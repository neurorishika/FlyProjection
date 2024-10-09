import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'basler_camera',
        'camera',
        'spinnaker_camera',
    },
    submod_attrs={
        'basler_camera': [
            'BaslerCamera',
            'camera_max_fps',
            'list_basler_cameras',
            'video_writer',
        ],
        'camera': [
            'CameraError',
            'SpinnakerCamera',
            'list_cameras',
            'video_writer',
        ],
        'spinnaker_camera': [
            'CameraError',
            'SpinnakerCamera',
            'list_spinnaker_cameras',
            'video_writer',
        ],
    },
)

__all__ = ['BaslerCamera', 'CameraError', 'SpinnakerCamera', 'basler_camera',
           'camera', 'camera_max_fps', 'list_basler_cameras', 'list_cameras',
           'list_spinnaker_cameras', 'spinnaker_camera', 'video_writer']
