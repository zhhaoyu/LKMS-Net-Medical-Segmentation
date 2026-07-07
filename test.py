import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.model import MF2_Net


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test MF2-Net on Medical Segmentation Datasets'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to trained model weights (.hdf5)'
    )
    parser.add_argument(
        '--image_path', type=str, required=True,
        help='Path to test image'
    )
    parser.add_argument(
        '--mask_path', type=str, default=None,
        help='Path to ground truth mask (optional)'
    )
    parser.add_argument(
        '--save_path', type=str, default=None,
        help='Path to save prediction result'
    )
    parser.add_argument(
        '--input_size', type=int, nargs=2, default=[256, 256],
        help='Input image size (height width)'
    )
    return parser.parse_args()


def preprocess_image(img_path, target_size=(256, 256)):
    img = load_img(img_path)
    img = img.resize(target_size)
    img = img_to_array(img).astype('float32')
    img = (img - img.mean()) / (img.std() + 1e-8)
    img = np.expand_dims(img, axis=0)
    return img


def postprocess_prediction(pred):
    pred = pred.squeeze()
    pred = (pred > 0.5).astype(np.uint8) * 255
    return pred


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    print(f"Loading model from: {args.model_path}")
    model = MF2_Net(input_shape=(args.input_size[0], args.input_size[1], 3))
    model.load_weights(args.model_path)

    print(f"Processing image: {args.image_path}")
    img = preprocess_image(args.image_path, tuple(args.input_size))

    pred = model.predict(img)
    pred_mask = postprocess_prediction(pred)

    if args.save_path:
        from PIL import Image
        Image.fromarray(pred_mask).save(args.save_path)
        print(f"Prediction saved to: {args.save_path}")

    if args.mask_path:
        from src.metrics import dice_coefficient, iou, f1_score
        import tensorflow as tf

        gt_mask = load_img(args.mask_path, color_mode='grayscale')
        gt_mask = gt_mask.resize(tuple(args.input_size))
        gt_mask = img_to_array(gt_mask).astype('float32') / 255.0
        gt_mask = (gt_mask > 0.5).astype(np.float32)

        pred_binary = (pred.squeeze() > 0.5).astype(np.float32)

        dice = dice_coefficient(gt_mask, pred_binary).numpy()
        iou_val = iou(gt_mask, pred_binary).numpy()
        f1_val = f1_score(gt_mask, pred_binary).numpy()

        print("\n" + "=" * 40)
        print("Evaluation Results:")
        print(f"Dice Coefficient: {dice:.4f}")
        print(f"IoU: {iou_val:.4f}")
        print(f"F1 Score: {f1_val:.4f}")
        print("=" * 40)


if __name__ == '__main__':
    main()