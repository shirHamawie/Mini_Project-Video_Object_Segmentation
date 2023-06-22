from video_object_segmentation import VideoObjectSegmentation
import sys


def main(*args, **kwargs):
    name = kwargs.get("name")
    if not name:
        raise ValueError("name must be provided")
    init_params = {}
    segments_params = {}

    for key, value in kwargs.items():
        if key in ["name", "data_path"]:
            init_params[key] = value
        elif key in ["include_xy", "complement_with_white"]:
            init_params[key] = bool(value)
        elif key in ["resize_ratio"]:
            init_params[key] = float(value)
        elif key in ["use_max_pdf", "use_niw_prior", "run_fit_partial"]:
            segments_params[key] = bool(value)
        elif key in ["iters_of_fit_partial", "frames_num", "verbose"]:
            segments_params[key] = int(value)
        elif key in ["close_tol", "redundancy_tol", "alpha", "epsilon"]:
            segments_params[key] = float(value)
        else:
            raise ValueError(f"Unknown key: {key}")

    vos = VideoObjectSegmentation(**init_params)
    vos.segment(**segments_params)


if __name__ == '__main__':
    # TODO: uncomment this to use command line arguments, fix 'name' parameter
    # args = sys.argv[1:]
    # kwargs = {}
    # for arg in args:
    #     if arg.startswith("--"):
    #         key, value = arg.split("=")
    #         kwargs[key[2:]] = value
    # main(*args, **kwargs)

    dinosaur_config1 = {
        "name": "dinosaur",
        "include_xy": True,
        "resize_ratio": 0.5,
        "complement_with_white": False,
        "frames_num": 24 * 5,
        "verbose": 1,
        "use_max_pdf": False,
        "alpha": 100.0,
        "epsilon": 0.0000001,
        "use_niw_prior": True,
        "run_fit_partial": True,
        "iters_of_fit_partial": 5,
        "close_tol": 1e-8,
        "redundancy_tol": 1e-8,
    }

    dinosaur_config2 = {
        "name": "dinosaur",
        "include_xy": True,
        "resize_ratio": 0.5,
        "complement_with_white": False,
        "frames_num": 24 * 5,
        "verbose": 1,
        "use_max_pdf": False,
        "alpha": 100.0,
        "epsilon": 0.0000001,
        "use_niw_prior": True,
        "run_fit_partial": True,
        "iters_of_fit_partial": 3,
        "close_tol": 1e-9,
        "redundancy_tol": 1e-9,
    }

    swan_config1 = {
        "name": "swan",
        "include_xy": True,
        "resize_ratio": 0.5,
        "complement_with_white": False,
        "frames_num": 24 * 5,
        "verbose": 1,
        "use_max_pdf": False,
        "alpha": 100.0,
        "epsilon": 0.0000001,
        "use_niw_prior": True,
        "run_fit_partial": True,
        "iters_of_fit_partial": 3,
        "close_tol": 1e-7,
        "redundancy_tol": 1e-7,
    }

    giraffe_config1 = {
        "name": "giraffe",
        "include_xy": True,
        "resize_ratio": 0.5,
        "complement_with_white": False,
        "frames_num": 24 * 5,
        "verbose": 1,
        "use_max_pdf": False,
        "alpha": 100.0,
        "epsilon": 0.0000001,
        "use_niw_prior": True,
        "run_fit_partial": True,
        "iters_of_fit_partial": 3,
        "close_tol": 1e-7,
        "redundancy_tol": 1e-7,
    }

    main(**giraffe_config1)


