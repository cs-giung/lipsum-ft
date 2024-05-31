import copy
import datetime
import itertools
import math
import os
import sys
import warnings
from collections import OrderedDict
from functools import partial
sys.path.append('./') # pylint: disable=wrong-import-position

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import flax
import optax
import transformers
import tensorflow_datasets as tfds
from flax import jax_utils, serialization
from flax.training import checkpoints, common_utils, train_state
from tabulate import tabulate
from tensorflow.io.gfile import GFile
from transformers import FlaxCLIPVisionModel, FlaxCLIPTextModel
from tqdm import tqdm

from scripts import defaults
from src import input_pipeline
from src.data import build_dataloader, PROJECT_LOGITS_FN
from src.metrics import evaluate_acc, evaluate_nll


def launch(config, print_fn):

    local_device_count = jax.local_device_count()
    shard_shape = (local_device_count, -1)

    # ----------------------------------------------------------------------- #
    # Dataset
    # ----------------------------------------------------------------------- #
    def prepare_tf_data(batch):
        batch['images'] = batch['images']._numpy() / 255.0
        batch['labels'] = batch['labels']._numpy()
        batch['marker'] = np.ones_like(batch['labels'])
        def _prepare(x):
            if x.shape[0] < config.batch_size:
                x = np.concatenate([x, np.zeros([
                    config.batch_size - x.shape[0], *x.shape[1:]
                ], x.dtype)])
            return x.reshape(shard_shape + x.shape[1:])
        return jax.tree_util.tree_map(_prepare, batch)

    dataset_builder = tfds.builder(config.data_name)

    trn_split = 'train'
    trn_steps_per_epoch = math.ceil(
        dataset_builder.info.splits[trn_split].num_examples / config.batch_size)
    trn_iter = map(prepare_tf_data, input_pipeline.create_trn_split(
        dataset_builder, config.batch_size, split=trn_split))
    trn_iter = jax_utils.prefetch_to_device(trn_iter, config.prefetch_factor)

    if config.data_name == 'imagenet2012':
        val_split = 'validation'
        NUM_CLASSES = 1000
    if config.data_name == 'domainnet':
        val_split = 'test'
        NUM_CLASSES = 345
    val_steps_per_epoch = math.ceil(
        dataset_builder.info.splits[val_split].num_examples / config.batch_size)
    val_iter = map(prepare_tf_data, input_pipeline.create_val_split(
        dataset_builder, config.batch_size, split=val_split))
    val_iter = jax_utils.prefetch_to_device(val_iter, config.prefetch_factor)

    # ----------------------------------------------------------------------- #
    # Model
    # ----------------------------------------------------------------------- #
    model = FlaxCLIPVisionModel.from_pretrained(
        pretrained_model_name_or_path=config.clip_model,
        dtype=jnp.dtype(config.dtype))

    # load pre-trained parameters
    pretrained_params = copy.deepcopy(model.params)
    pretrained_params['visual_projection'] = checkpoints.restore_checkpoint(
        f'./save/clip/{config.clip_model}/visual_projection.ckpt', target=None
    )['kernel']
    pretrained_params['logit_scale'] = checkpoints.restore_checkpoint(
        f'./save/clip/{config.clip_model}/logit_scale.ckpt', target=None)

    # define forward function and specify shapes
    pixel_mean = jnp.array([
        0.48145466, 0.45782750, 0.40821073]).reshape(1, 3, 1, 1)
    pixel_std = jnp.array([
        0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)

    def get_features(images, params):
        images = (images.transpose(0, 3, 1, 2) - pixel_mean) / pixel_std
        pooler_output = model(images, params).pooler_output
        proj_features = pooler_output @ params['visual_projection']
        norm_features = jnp.linalg.norm(proj_features, axis=-1, keepdims=True)
        norm_features = proj_features / norm_features
        return norm_features

    images = next(trn_iter)['images']
    output = jax.pmap(get_features)(
        images, jax_utils.replicate(pretrained_params))
    FEATURE_DIM = output.shape[-1]

    log_str = f'images.shape: {images.shape}, output.shape: {output.shape}'
    print_fn(log_str)

    # define initial_ext_params
    if config.clip_ext_init is None:
        initial_ext_params = pretrained_params
    elif config.clip_ext_init.endswith('.ckpt'):
        initial_ext_params = checkpoints.restore_checkpoint(
            config.clip_ext_init, target=None)['ext']
    else:
        raise NotImplementedError(
            f'unknown config.clip_ext_init={config.clip_ext_init}')

    # define initial_cls_params
    if config.clip_cls_init is None:
        initial_cls_params = jnp.zeros((FEATURE_DIM, NUM_CLASSES))
    elif config.clip_cls_init.endswith('.ckpt'):
        initial_cls_params = checkpoints.restore_checkpoint(
            config.clip_cls_init, target=None)['cls']
    else:
        raise NotImplementedError(
            f'unknown config.clip_cls_init={config.clip_cls_init}')

    # setup trainable parameters
    params = {'cls': initial_cls_params, 'ext': initial_ext_params}
    log_str = 'The number of trainable parameters: {:d}'.format(
        jax.flatten_util.ravel_pytree(params)[0].size)
    print_fn(log_str)

    # setup text model
    clip_text_model = FlaxCLIPTextModel.from_pretrained(
        pretrained_model_name_or_path=config.clip_model,
        dtype=jnp.dtype(config.dtype))
    clip_text_projection = checkpoints.restore_checkpoint(
        f'./save/clip/{config.clip_model}/text_projection.ckpt', target=None
    )['kernel']

    # ----------------------------------------------------------------------- #
    # Optimization
    # ----------------------------------------------------------------------- #
    def step_trn(state, batch, config, scheduler, text_rng):

        _, new_text_rng = jax.random.split(text_rng)

        def _global_norm(updates):
            return jnp.sqrt(sum(jnp.sum(
                jnp.square(e)) for e in jax.tree_util.tree_leaves(updates)))

        def _clip_by_global_norm(updates, global_norm):
            return jax.tree_util.tree_map(
                lambda e: jnp.where(
                    global_norm < config.optim_global_clipping, e,
                    (e / global_norm) * config.optim_global_clipping), updates)

        # define loss function
        def loss_fn(params):

            # get txt features
            rngs = jax.random.split(text_rng, config.token_k_ways)
            txts = [jnp.array(
                [49406,] + [0,] * config.token_length + [49407,]).at[
                    1:1+config.token_length
                ].set(jax.random.randint(
                    rngs[iii], (config.token_length,), minval=0, maxval=49406))
                for iii in range(config.token_k_ways)]
            txts = clip_text_model(jnp.stack(txts)).pooler_output
            txts = txts @ clip_text_projection
            txts = txts / jnp.linalg.norm(txts, axis=-1, keepdims=True)

            # get features
            priors = get_features(batch['images'], initial_ext_params)
            output = get_features(batch['images'], params['ext'])

            # negative_log_likelihood
            smooth = config.optim_label_smoothing
            target = common_utils.onehot(batch['labels'], NUM_CLASSES)
            target = (1.0 - smooth) * target + \
                smooth * jnp.ones_like(target) / NUM_CLASSES
            logits = jnp.exp(
                pretrained_params['logit_scale']) * output @ params['cls']
            source = jax.nn.log_softmax(logits, axis=-1)
            negative_log_likelihood = -jnp.sum(target * source, axis=-1)
            negative_log_likelihood = jnp.mean(negative_log_likelihood)

            # feat_regularization
            target = jnp.exp(
                pretrained_params['logit_scale']) * priors @ txts.T
            source = jnp.exp(
                pretrained_params['logit_scale']) * output @ txts.T

            feat_regularization = jnp.sum(
                jnp.square(source - target), axis=-1
                ) / (2 * config.token_k_ways)
            feat_regularization = jnp.mean(feat_regularization)

            # loss
            loss = negative_log_likelihood \
                + config.feat_regularization * feat_regularization

            # log metrics
            metrics = OrderedDict({
                'loss': loss,
                'negative_log_likelihood': negative_log_likelihood,
                'feat_regularization': feat_regularization})
            return loss, metrics

        # compute losses and gradients
        aux, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name='batch')

        # compute norms of weights and gradients
        w_norm = _global_norm(state.params)
        g_norm = _global_norm(grads)
        if config.optim_global_clipping:
            grads = _clip_by_global_norm(grads, g_norm)

        # get auxiliaries
        metrics = jax.lax.pmean(aux[1], axis_name='batch')
        metrics['w_norm'] = w_norm
        metrics['g_norm'] = g_norm
        metrics['lr'] = scheduler(state.step)

        # update train state
        new_state = state.apply_gradients(grads=grads)
        return new_state, metrics, new_text_rng

    # define optimizer with scheduler
    scheduler = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value       = 0.0,
                end_value        = config.optim_lr,
                transition_steps = math.floor(0.1 * config.optim_ni)),
            optax.cosine_decay_schedule(
                init_value       = config.optim_lr,
                decay_steps      = math.floor(0.9 * config.optim_ni))
        ], boundaries=[
            math.floor(0.1 * config.optim_ni),
        ])
    optimizer = optax.adamw(
        scheduler, b1=config.optim_b1, b2=config.optim_b2,
        eps=config.optim_eps, eps_root=config.optim_eps_root,
        mu_dtype=jnp.dtype(config.dtype),
        weight_decay=config.optim_weight_decay)

    # build and replicate train state
    def apply_fn(images, params):
        output = get_features(images, params['ext'])
        logits = jnp.exp(
            pretrained_params['logit_scale']) * output @ params['cls']
        return logits

    state = train_state.TrainState.create(
        apply_fn=apply_fn, params=params, tx=optimizer)
    state = jax_utils.replicate(state)

    # run optimization
    best_acc = 0.0
    p_step_trn = jax.pmap(partial(
        step_trn, config=config, scheduler=scheduler), axis_name='batch')
    p_apply_fn = jax.pmap(apply_fn, axis_name='batch')
    text_rng = jax_utils.replicate(jax.random.PRNGKey(config.seed))

    trn_metric = []
    for iter_idx in itertools.count(start=1):

        # rendezvous
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

        # terminate training
        if iter_idx == config.optim_ni + 1:
            break

        # ------------------------------------------------------------------- #
        # Train
        # ------------------------------------------------------------------- #
        log_str = '[Iter {:7d}/{:7d}] '.format(iter_idx, config.optim_ni)

        batch = next(trn_iter)
        state, metrics, text_rng = p_step_trn(state, batch, text_rng=text_rng)
        trn_metric.append(metrics)

        if iter_idx % 1000 == 0:

            trn_metric = common_utils.get_metrics(trn_metric)
            trn_summarized = {f'trn/{k}': v for k, v in jax.tree_util.tree_map(
                lambda e: e.mean(), trn_metric).items()}
            trn_metric = []

            log_str += ', '.join(
                f'{k} {v:.3e}' for k, v in trn_summarized.items())

            # --------------------------------------------------------------- #
            # Valid
            # --------------------------------------------------------------- #
            val_summarized = {}
            acc, nll, cnt = 0.0, 0.0, 0
            for batch_idx, batch in enumerate(val_iter, start=1):
                logits = p_apply_fn(batch['images'], state.params)
                logits = logits.reshape(-1, NUM_CLASSES)
                labels = batch['labels'].reshape(-1)
                marker = batch['marker'].reshape(-1)
                log_confidences = jax.nn.log_softmax(logits, axis=-1)
                acc += jnp.sum(jnp.where(marker, evaluate_acc(
                    log_confidences, labels, log_input=True, reduction='none'),
                    marker))
                nll += jnp.sum(jnp.where(marker, evaluate_nll(
                    log_confidences, labels, log_input=True, reduction='none'),
                    marker))
                cnt += jnp.sum(marker)
                if batch_idx == val_steps_per_epoch:
                    break
            val_summarized['val/acc'] = acc / cnt
            val_summarized['val/nll'] = nll / cnt
            val_summarized['val/best_acc'] = max(
                val_summarized['val/acc'], best_acc)

            log_str += ', '
            log_str += ', '.join(
                f'{k} {v:.3e}' for k, v in val_summarized.items())

            # --------------------------------------------------------------- #
            # Save
            # --------------------------------------------------------------- #
            tst_summarized = {}
            if best_acc < val_summarized['val/acc']:

                log_str += ' (best_acc: {:.3e} -> {:.3e})'.format(
                    best_acc, val_summarized['val/acc'])
                best_acc = val_summarized['val/acc']
                best_params = jax.device_get(
                    jax.tree_util.tree_map(lambda x: x[0], state.params))

                if config.save:
                    best_path = os.path.join(config.save, 'best_acc.ckpt')
                    with GFile(best_path, 'wb') as fp:
                        fp.write(serialization.to_bytes(best_params))

            # logging current iteration
            print_fn(log_str)

            # terminate training if loss is nan
            if jnp.isnan(trn_summarized['trn/loss']):
                break

    # ----------------------------------------------------------------------- #
    # Test distribution shifts
    # ----------------------------------------------------------------------- #
    tst_summarized = {}
    best_params = jax_utils.replicate(best_params)

    if config.data_name == 'imagenet2012':
        for dataset_name in ['ImageNet', 'ImageNetV2',
                             'ImageNetR', 'ImageNetA', 'ImageNetSketch']:

            tst_images = np.load(os.path.join(
                config.data_root, f'{dataset_name}_x224/test_images.npy'))
            tst_labels = np.load(os.path.join(
                config.data_root, f'{dataset_name}_x224/test_labels.npy'))
            dataloader = build_dataloader(
                tst_images, tst_labels, config.batch_size)
            tst_steps_per_epoch = math.ceil(
                tst_images.shape[0] / config.batch_size)

            acc, nll, cnt = 0.0, 0.0, 0
            for batch in tqdm(dataloader, leave=False, ncols=0,
                            total=tst_steps_per_epoch, desc=dataset_name):

                images = jnp.array(batch['images'] / 255.0)
                if images.shape[0] != config.batch_size:
                    images = jnp.zeros(
                        (config.batch_size,) + images.shape[1:]
                    ).at[:images.shape[0]].set(images)
                images = images.reshape(shard_shape + images.shape[1:])

                labels = jnp.array(batch['labels'])[:batch['marker'].sum()]
                logits = p_apply_fn(images, best_params)
                logits = logits.reshape(-1, NUM_CLASSES)[:batch['marker'].sum()]
                project_logits = PROJECT_LOGITS_FN[dataset_name]
                if project_logits is not None:
                    logits = project_logits(logits)

                pre = jax.nn.softmax(logits, axis=-1)
                acc += evaluate_acc(
                    pre, labels, log_input=False, reduction='sum')
                nll += evaluate_nll(
                    pre, labels, log_input=False, reduction='sum')
                cnt += labels.shape[0]

            tst_summarized[f'{dataset_name}/acc'] = acc / cnt
            tst_summarized[f'{dataset_name}/nll'] = nll / cnt

    if config.data_name == 'domainnet':
        for dataset_name in ['DomainNetReal', 'DomainNetPainting',
                             'DomainNetClipart', # 'DomainNetQuickdraw',
                             'DomainNetInfograph', 'DomainNetSketch']:

            tst_images = np.load(os.path.join(
                config.data_root, f'{dataset_name}_x224/test_images.npy'))
            tst_labels = np.load(os.path.join(
                config.data_root, f'{dataset_name}_x224/test_labels.npy'))
            dataloader = build_dataloader(
                tst_images, tst_labels, config.batch_size)
            tst_steps_per_epoch = math.ceil(
                tst_images.shape[0] / config.batch_size)

            acc, nll, cnt = 0.0, 0.0, 0
            for batch in tqdm(dataloader, leave=False, ncols=0,
                            total=tst_steps_per_epoch, desc=dataset_name):

                images = jnp.array(batch['images'] / 255.0)
                if images.shape[0] != config.batch_size:
                    images = jnp.zeros(
                        (config.batch_size,) + images.shape[1:]
                    ).at[:images.shape[0]].set(images)
                images = images.reshape(shard_shape + images.shape[1:])

                labels = jnp.array(batch['labels'])[:batch['marker'].sum()]
                logits = p_apply_fn(images, best_params)
                logits = logits.reshape(-1, NUM_CLASSES)[:batch['marker'].sum()]

                pre = jax.nn.softmax(logits, axis=-1)
                acc += evaluate_acc(
                    pre, labels, log_input=False, reduction='sum')
                nll += evaluate_nll(
                    pre, labels, log_input=False, reduction='sum')
                cnt += labels.shape[0]

            tst_summarized[f'{dataset_name}/acc'] = acc / cnt
            tst_summarized[f'{dataset_name}/nll'] = nll / cnt

    # logging current iteration
    log_str = ', '.join(f'{k} {v:.3e}' for k, v in tst_summarized.items())
    print_fn(log_str)


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    parser.add_argument(
        '--feat_regularization', default=1.0, type=float,
        help='regularizing coefficient (default: 1.0)')
    parser.add_argument(
        '--token_length', default=8, type=int,
        help='length of random tokens (default: 8)')
    parser.add_argument(
        '--token_k_ways', default=80, type=int,
        help='length of random tokens (default: 80)')

    parser.add_argument(
        '--optim_ni', default=50000, type=int,
        help='the number of training iterations (default: 50000)')
    parser.add_argument(
        '--optim_lr', default=1e-05, type=float,
        help='base learning rate (default: 1e-05)')
    parser.add_argument(
        '--optim_b1', default=0.9, type=float,
        help='rate for the first moment of past gradients (default: 0.9)')
    parser.add_argument(
        '--optim_b2', default=0.999, type=float,
        help='rate for the second moment of past gradients (default: 0.999)')
    parser.add_argument(
        '--optim_eps', default=1e-08, type=float,
        help='epsilon value outside of the square root (default: 1e-08)')
    parser.add_argument(
        '--optim_eps_root', default=0.0, type=float,
        help='epsilon value inside of the square root (default: 0.0)')
    parser.add_argument(
        '--optim_weight_decay', default=0.0, type=float,
        help='weight decay coefficient (default: 0.0)')

    parser.add_argument(
        '--optim_label_smoothing', default=0.0, type=float,
        help='label smoothing regularization (default: 0.0)')
    parser.add_argument(
        '--optim_global_clipping', default=None, type=float,
        help='global norm for the gradient clipping (default: None)')

    parser.add_argument(
        '--save', default=None, type=str,
        help='save the *.log and *.ckpt files if specified (default: False)')
    parser.add_argument(
        '--seed', default=None, type=int,
        help='random seed for training (default: None)')

    parser.add_argument(
        '--dtype', default='bfloat16', type=str,
        help='dtype of computation and accumulator (default: bfloat16)')

    args = parser.parse_args()

    if args.seed is None:
        args.seed = (
            os.getpid()
            + int(datetime.datetime.now().strftime('%S%f'))
            + int.from_bytes(os.urandom(2), 'big')
        )

    if args.save == 'auto':
        args.save = os.path.join(
            f'./save/clip/{args.clip_model}/{args.data_name}/AdamW-FeatRegRandomText-MSE/',
            f'{TIME_STAMP}_{args.seed}')

    if args.save is not None:
        if os.path.exists(args.save):
            raise AssertionError(f'already existing args.save = {args.save}')
        os.makedirs(args.save, exist_ok=True)

    def print_fn(s):
        s = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + s
        if args.save is not None:
            with open(os.path.join(args.save, f'{TIME_STAMP}.log'), 'a') as fp:
                fp.write(s + '\n')
        print(s, flush=True)

    log_str = tabulate([
        ('sys.platform', sys.platform),
        ('Python', sys.version.replace('\n', '')),
        ('JAX', jax.__version__
            + ' @' + os.path.dirname(jax.__file__)),
        ('jaxlib', jaxlib.__version__
            + ' @' + os.path.dirname(jaxlib.__file__)),
        ('Flax', flax.__version__
            + ' @' + os.path.dirname(flax.__file__)),
        ('Optax', optax.__version__
            + ' @' + os.path.dirname(optax.__file__)),
        ('Transformers', transformers.__version__
            + ' @' + os.path.dirname(transformers.__file__)),
    ]) + '\n'
    log_str = f'Environments:\n{log_str}'
    print_fn(log_str)

    log_str = ''
    max_k_len = max(map(len, vars(args).keys()))
    for k, v in vars(args).items():
        log_str += f'- args.{k.ljust(max_k_len)} : {v}\n'
    log_str = f'Command line arguments:\n{log_str}'
    print_fn(log_str)

    if jax.local_device_count() > 1:
        log_str = (
            'Multiple local devices are detected:\n' f'{jax.local_devices()}\n')
        print_fn(log_str)

    launch(args, print_fn)


if __name__ == '__main__':
    main()
