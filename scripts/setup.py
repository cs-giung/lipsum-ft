import copy
import datetime
import math
import os
import sys
from collections import OrderedDict
sys.path.append('./') # pylint: disable=wrong-import-position

import flax
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax
import transformers
from flax import serialization
from tabulate import tabulate
from tensorflow.io.gfile import GFile
from transformers import FlaxCLIPModel, CLIPTokenizer
from tqdm import tqdm

from scripts import defaults, classnames, templates
from src.data import build_dataloader, PROJECT_LOGITS_FN
from src.metrics import evaluate_acc, evaluate_nll


def launch(config, print_fn):
    """Creates zero-shot weights."""
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    local_device_count = jax.local_device_count()
    shard_shape = (local_device_count, -1)

    # ----------------------------------------------------------------------- #
    # Model
    # ----------------------------------------------------------------------- #

    # build the whole model with pre-trained parameters
    model = FlaxCLIPModel.from_pretrained(config.clip_model)
    pretrained_params = copy.deepcopy(model.params)

    # save pre-trained parameters
    ckpt_path = os.path.join(
        f'./save/clip/{config.clip_model}', 'logit_scale.ckpt')
    with GFile(ckpt_path, 'wb') as fp:
        fp.write(serialization.to_bytes(
            pretrained_params['logit_scale']))

    ckpt_path = os.path.join(
        f'./save/clip/{config.clip_model}', 'visual_projection.ckpt')
    with GFile(ckpt_path, 'wb') as fp:
        fp.write(serialization.to_bytes(
            pretrained_params['visual_projection']))

    ckpt_path = os.path.join(
        f'./save/clip/{config.clip_model}', 'text_projection.ckpt')
    with GFile(ckpt_path, 'wb') as fp:
        fp.write(serialization.to_bytes(
            pretrained_params['text_projection']))

    # define tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(config.clip_model)
    get_text_features = jax.jit(model.get_text_features)

    # ---------------------------------------------------------------------- #
    # Zeroshot (imagenet2012)
    # ---------------------------------------------------------------------- #
    if config.data_name == 'imagenet2012':
        log_str = 'Create zeroshot weights for imagenet2012...'
        print_fn(log_str)

        texts = []
        for c in tqdm(classnames.openai_imagenet_classnames, leave=False):
            for t in getattr(templates, config.template):
                texts.append(t(c))
        texts = tokenizer(texts, padding=True, return_tensors='np')
        num_cls = len(classnames.openai_imagenet_classnames)
        num_ctx = len(getattr(templates, config.template))

        # create zeroshot weights
        zeroshot_raw_weights = []
        for idx in tqdm(range(num_cls), leave=False):
            input_ids = texts.input_ids.reshape(num_cls, num_ctx, -1)[idx, :, :]
            attn = texts.attention_mask.reshape(num_cls, num_ctx, -1)[idx, :, :]
            feats = get_text_features(input_ids, attn, params=pretrained_params)
            feats = feats / jnp.linalg.norm(feats, axis=-1, keepdims=True)
            zeroshot_raw_weights.append(feats)
        zeroshot_raw_weights = jnp.stack(zeroshot_raw_weights, axis=0).T
        log_str = f'zeroshot_raw_weights.shape: {zeroshot_raw_weights.shape}'
        print_fn(log_str)

        zeroshot_cls_weights = jnp.mean(zeroshot_raw_weights, axis=1)
        zeroshot_cls_weights = zeroshot_cls_weights / jnp.linalg.norm(
            zeroshot_cls_weights, axis=0, keepdims=True)
        log_str = f'zeroshot_cls_weights.shape: {zeroshot_cls_weights.shape}'
        print_fn(log_str)

        zeroshot_ctx_weights = jnp.mean(zeroshot_raw_weights, axis=2)
        zeroshot_ctx_weights = zeroshot_ctx_weights / jnp.linalg.norm(
            zeroshot_ctx_weights, axis=0, keepdims=True)
        log_str = f'zeroshot_ctx_weights.shape: {zeroshot_ctx_weights.shape}'
        print_fn(log_str)

        # save zeroshot weights
        ckpt_path = os.path.join(config.save, 'imagenet2012.ckpt')
        with GFile(ckpt_path, 'wb') as fp:
            fp.write(serialization.to_bytes({
                'cls': zeroshot_cls_weights,
                'ctx': zeroshot_ctx_weights,
                'raw': zeroshot_raw_weights,}))

    # evaluate zeroshot weights
    if config.data_name == 'imagenet2012' and config.evaluate:
        tst_summarized = OrderedDict()

        pixel_mean = jnp.array([
            0.48145466, 0.45782750, 0.40821073]).reshape(1, 3, 1, 1)
        pixel_std = jnp.array([
            0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)

        def preprocess_images(images):
            images = jnp.array(images.transpose((0, 3, 1, 2)) / 255.0)
            return (images - pixel_mean) / pixel_std

        @jax.pmap
        def p_apply_fn(images):
            feat = model.get_image_features(images, pretrained_params)
            feat = feat / jnp.linalg.norm(feat, axis=-1, keepdims=True)
            temp = jnp.exp(pretrained_params['logit_scale'])
            return feat @ zeroshot_cls_weights * temp

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

                images = preprocess_images(batch['images'])
                if images.shape[0] != config.batch_size:
                    images = jnp.zeros(
                        (config.batch_size,) + images.shape[1:]
                    ).at[:images.shape[0]].set(images)
                images = images.reshape(shard_shape + images.shape[1:])

                labels = jnp.array(batch['labels'])[:batch['marker'].sum()]
                logits = p_apply_fn(images)
                logits = logits.reshape(-1, num_cls)[:batch['marker'].sum()]
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

        log_str = ', '.join(
            f'{k} {v:.3e}' for k, v in tst_summarized.items()) + '\n'
        print_fn(log_str)

    # ---------------------------------------------------------------------- #
    # Zeroshot (domainnet)
    # ---------------------------------------------------------------------- #
    if config.data_name == 'domainnet':
        log_str = 'Create zeroshot weights for domainnet...'
        print_fn(log_str)

        texts = []
        for c in tqdm(classnames.tfds_domainnet_classnames, leave=False):
            for t in getattr(templates, config.template):
                texts.append(t(c))
        texts = tokenizer(texts, padding=True, return_tensors='np')
        num_cls = len(classnames.tfds_domainnet_classnames)
        num_ctx = len(getattr(templates, config.template))

        # create zeroshot weights
        zeroshot_raw_weights = []
        for idx in tqdm(range(num_cls), leave=False):
            input_ids = texts.input_ids.reshape(num_cls, num_ctx, -1)[idx, :, :]
            attn = texts.attention_mask.reshape(num_cls, num_ctx, -1)[idx, :, :]
            feats = get_text_features(input_ids, attn, params=pretrained_params)
            feats = feats / jnp.linalg.norm(feats, axis=-1, keepdims=True)
            zeroshot_raw_weights.append(feats)
        zeroshot_raw_weights = jnp.stack(zeroshot_raw_weights, axis=0).T
        log_str = f'zeroshot_raw_weights.shape: {zeroshot_raw_weights.shape}'
        print_fn(log_str)

        zeroshot_cls_weights = jnp.mean(zeroshot_raw_weights, axis=1)
        zeroshot_cls_weights = zeroshot_cls_weights / jnp.linalg.norm(
            zeroshot_cls_weights, axis=0, keepdims=True)
        log_str = f'zeroshot_cls_weights.shape: {zeroshot_cls_weights.shape}'
        print_fn(log_str)

        zeroshot_ctx_weights = jnp.mean(zeroshot_raw_weights, axis=2)
        zeroshot_ctx_weights = zeroshot_ctx_weights / jnp.linalg.norm(
            zeroshot_ctx_weights, axis=0, keepdims=True)
        log_str = f'zeroshot_ctx_weights.shape: {zeroshot_ctx_weights.shape}'
        print_fn(log_str)

        # save zeroshot weights
        ckpt_path = os.path.join(config.save, 'domainnet.ckpt')
        with GFile(ckpt_path, 'wb') as fp:
            fp.write(serialization.to_bytes({
                'cls': zeroshot_cls_weights,
                'ctx': zeroshot_ctx_weights,
                'raw': zeroshot_raw_weights,}))

    # evaluate zeroshot weights
    if config.data_name == 'domainnet' and config.evaluate:
        tst_summarized = OrderedDict()

        pixel_mean = jnp.array([
            0.48145466, 0.45782750, 0.40821073]).reshape(1, 3, 1, 1)
        pixel_std = jnp.array([
            0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)

        def preprocess_images(images):
            images = jnp.array(images.transpose((0, 3, 1, 2)) / 255.0)
            return (images - pixel_mean) / pixel_std

        @jax.pmap
        def p_apply_fn(images):
            feat = model.get_image_features(images, pretrained_params)
            feat = feat / jnp.linalg.norm(feat, axis=-1, keepdims=True)
            temp = jnp.exp(pretrained_params['logit_scale'])
            return feat @ zeroshot_cls_weights * temp

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

                images = preprocess_images(batch['images'])
                if images.shape[0] != config.batch_size:
                    images = jnp.zeros(
                        (config.batch_size,) + images.shape[1:]
                    ).at[:images.shape[0]].set(images)
                images = images.reshape(shard_shape + images.shape[1:])

                labels = jnp.array(batch['labels'])[:batch['marker'].sum()]
                logits = p_apply_fn(images)
                logits = logits.reshape(-1, num_cls)[:batch['marker'].sum()]

                pre = jax.nn.softmax(logits, axis=-1)
                acc += evaluate_acc(
                    pre, labels, log_input=False, reduction='sum')
                nll += evaluate_nll(
                    pre, labels, log_input=False, reduction='sum')
                cnt += labels.shape[0]

            tst_summarized[f'{dataset_name}/acc'] = acc / cnt
            tst_summarized[f'{dataset_name}/nll'] = nll / cnt

        log_str = ', '.join(
            f'{k} {v:.3e}' for k, v in tst_summarized.items()) + '\n'
        print_fn(log_str)


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()
    parser.add_argument(
        '--template', default='openai_imagenet_template', type=str,
        help='specify prompt template (default: openai_imagenet_template)')
    parser.add_argument(
        '--evaluate', default=False, type=defaults.str2bool,
        help='run evaluation if specified (default: False)')
    args = parser.parse_args()

    args.save = f'./save/clip/{args.clip_model}/{args.template}'
    os.makedirs(args.save, exist_ok=True)

    def print_fn(s):
        s = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + s
        if args.save is not None:
            with open(os.path.join(args.save, f'{TIME_STAMP}.log'),
                      'a', encoding='utf-8') as fp:
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
