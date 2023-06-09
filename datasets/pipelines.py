import clip
import random
from PIL import Image
from torchvision import transforms
from mmcls.datasets.builder import PIPELINES


@PIPELINES.register_module()
class CLIPPreprocess:
    """CLIP processor."""

    def __init__(self, arch):
        _, self.preprocess = clip.load(arch)

    def __call__(self, results):
        filename = results['img_info']['filename']
        with open(filename, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.preprocess(img)
        results['filename'] = filename
        results['img'] = img
        return results


@PIPELINES.register_module()
class PatchGastricPipeline:
    """Translated from histopathology-image-caption."""

    img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __init__(self, test_mode=False):
        self.test_mode = test_mode

    def __call__(self, results):
        if 'feature' in results:
            feature = results['feature']
            r = random.randint(1, len(feature))
            results['img'] = feature[r - 1]
            return results

        if not self.test_mode:
            transform = [
                transforms.Resize((224, 224)),  # seems original code does not contain resize
                # transforms.CenterCrop(224),  # try this?
                transforms.ColorJitter(brightness=0.2, saturation=(0, 0.2), hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)
            ]
            r = random.randint(0, 3)
            for _ in range(r):
                transform.append(transforms.RandomRotation((90, 90)))
            transform.extend([
                transforms.ToTensor(),
                transforms.Normalize(**self.img_norm)
            ])
        else:
            transform = [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(**self.img_norm)
            ]
        transform = transforms.Compose(transform)
        filename = results['img_info']['filename']
        with open(filename, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if 'img_pil' in results:
            img = results['img_pil']
        img = transform(img)
        results['filename'] = filename
        results['img'] = img
        return results
