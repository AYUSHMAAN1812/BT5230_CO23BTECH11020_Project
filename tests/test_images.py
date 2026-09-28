from PIL import Image

from ocsr_project.images import generate_depiction


def test_generate_clean_and_augmented_images(tmp_path):
    clean = generate_depiction("CCO", tmp_path / "clean.png", width=96, height=96)
    augmented = generate_depiction(
        "CCO",
        tmp_path / "augmented.png",
        condition="handdrawn_augmented",
        seed=11,
        width=96,
        height=96,
    )
    assert Image.open(clean).size == (96, 96)
    assert Image.open(augmented).size == (96, 96)
    assert clean.read_bytes() != augmented.read_bytes()

