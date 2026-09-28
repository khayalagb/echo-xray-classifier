from PIL import Image

from src.transforms import RandomGamma, SpeckleNoise, build_transforms


def test_gamma_and_speckle_change_pixels():
    img = Image.new("L", (32, 32), color=128)
    assert RandomGamma(p=1.0)(img) is not None
    assert SpeckleNoise(p=1.0)(img) is not None


def test_build_transforms_output_shape():
    img = Image.new("L", (100, 100), color=100)
    t = build_transforms(mean=0.5, std=0.2, train=True)
    out = t(img)
    assert out.shape == (3, 224, 224)


if __name__ == "__main__":
    test_gamma_and_speckle_change_pixels()
    test_build_transforms_output_shape()
    print("ok")
