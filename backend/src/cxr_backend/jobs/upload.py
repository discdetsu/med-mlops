"""Accept only what `cxr_core.preprocess.load_image` can read, recognized by content."""

from cxr_backend.jobs.models import InferenceJob

_PNG = b"\x89PNG\r\n\x1a\n"
_JPEG = b"\xff\xd8\xff"

CONTENT_TYPES = {
    InferenceJob.Format.PNG: "image/png",
    InferenceJob.Format.JPEG: "image/jpeg",
    InferenceJob.Format.DICOM: "application/dicom",
}
EXTENSIONS = {
    InferenceJob.Format.PNG: ".png",
    InferenceJob.Format.JPEG: ".jpg",
    InferenceJob.Format.DICOM: ".dcm",
}


def sniff_format(header: bytes) -> InferenceJob.Format | None:
    """`header` is the first 132 bytes or more. DICOM Part 10 has `DICM` at byte 128."""
    if header.startswith(_PNG):
        return InferenceJob.Format.PNG
    if header.startswith(_JPEG):
        return InferenceJob.Format.JPEG
    if header[128:132] == b"DICM":
        return InferenceJob.Format.DICOM
    return None
