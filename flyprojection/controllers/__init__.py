import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'camera',
    },
    submod_attrs={
        'camera': [
            'CameraError',
            'SpinnakerCamera',
            'list_cameras',
            'video_writer',
        ],
    },
)

__all__ = ['CameraError', 'SpinnakerCamera', 'camera', 'list_cameras',
           'video_writer']
