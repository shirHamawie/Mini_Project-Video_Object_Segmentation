import os
import cv2
import numpy as np
from grabcut import GrabCut
from julia.api import Julia
from dpmmpythonStreaming.dpmmwrapper import DPMMPython
from dpmmpythonStreaming.priors import niw
from gaussian_mixture import GaussianMixture
from tqdm import tqdm
jl = Julia(compiled_modules=False)


def extract_gaussian_data_and_create_mixture(model):
    clusters = model.group.local_clusters
    means = []
    covariances = []
    clusters_sizes = []
    for cluster in clusters:
        distribution = cluster.cluster_params.cluster_params.distribution
        means.append(distribution.μ)
        covariances.append(distribution.Σ)
        clusters_sizes.append(cluster.points_count)
    all_points_count = sum(clusters_sizes)
    weights = [size / all_points_count for size in clusters_sizes]
    return GaussianMixture(means, covariances, weights)


def choose_fps(prev_fps: int, frames_num: int) -> int:
    value = frames_num // 10
    if value < 10:
        return value
    return prev_fps


class VideoObjectSegmentation:
    def __init__(
            self,
            name: str,
            resize_ratio: float = 1.0,
            include_xy: bool = False,
            complement_with_white: bool = False,
            data_path: str = "data",
    ):
        self.name = name
        self.data_path = data_path
        self._make_data_dir(data_path)
        self.video_path = f"data/videos/{name}.mp4"
        self.resize_ratio = resize_ratio
        self.include_xy = include_xy
        self.complement_with_white = complement_with_white

        self.fg = None
        self.bg = None

        self.fgm = None
        self.bgm = None
        self.verbose = False
        self.save_images = False
        self.show_clusters = False
        self.use_max_pdf = False
        self.frame_idx = 0
        self.original_shape = None
        self.close_tol = None
        self.redundancy_tol = None

    def _make_data_dir(self, data_path: str):
        videos_path = os.path.join(data_path, "videos")
        images_path = os.path.join(data_path, f"images/{self.name}")
        new_videos_path = os.path.join(data_path, "new_videos")
        new_frames_path = os.path.join(data_path, f"new_frames/{self.name}")

        os.makedirs(videos_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(new_videos_path, exist_ok=True)
        os.makedirs(new_frames_path, exist_ok=True)

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, (0, 0), fx=self.resize_ratio, fy=self.resize_ratio)

    def _create_priors(self, use_niw: bool = False):
        if use_niw:
            fg_mean = np.mean(self.fg, axis=1)
            fg_cov = np.cov(self.fg)
            fg_prior = niw(1, fg_mean, self.fg.shape[0], fg_cov)
            bg_mean = np.mean(self.bg, axis=1)
            bg_cov = np.cov(self.bg)
            bg_prior = niw(1, bg_mean, self.bg.shape[0], bg_cov)
        else:
            fg_prior = DPMMPython.create_prior(self.fg.shape[0], 0, 100, 80, 120)
            bg_prior = DPMMPython.create_prior(self.bg.shape[0], 0, 100, 80, 120)
        return fg_prior, bg_prior

    def _classify_first_frame(self, frame: np.ndarray, transparency: float = 0.2) -> np.ndarray:
        if self.verbose:
            print("Recreating first frame with foreground coloring...")
        foreground_coloring = np.zeros(frame.shape)
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if np.array_equal(frame[i, j], self.fg[i, j]):
                    foreground_coloring[i, j] = [0, 255, 0]

        marked_frame = cv2.addWeighted(foreground_coloring, transparency, frame, 1 - transparency, 0, dtype=cv2.CV_8U)
        if self.verbose:
            print("Done.")
        if self.save_images:
            cv2.imwrite(f"data/images/{self.name}/frame{self.frame_idx}.png", marked_frame)
        return marked_frame

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        shape = frame.shape
        mod_frame = frame
        if self.include_xy:
            shape = (shape[0], shape[1], shape[2] + 2)
            mod_frame = np.zeros(shape)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    mod_frame[i, j] = np.append(frame[i, j], [i / shape[0], j / shape[1]])
        mod_frame = mod_frame.reshape(shape[0] * shape[1], shape[2]).astype(np.float32)
        mod_frame = mod_frame.T
        return mod_frame

    def _is_pixel_in_object(self, pixel: np.ndarray):
        eval_pdf = GaussianMixture.max_pdf if self.use_max_pdf else GaussianMixture.pdf
        fg_pdf_val = eval_pdf(self.fgm, pixel)
        bg_pdf_val = eval_pdf(self.bgm, pixel)
        close = np.isclose(fg_pdf_val, bg_pdf_val, atol=self.close_tol)
        redundant = fg_pdf_val < self.redundancy_tol
        return fg_pdf_val > bg_pdf_val and not close and not redundant

    def _classify_frame(
            self,
            origin_frame: np.ndarray,
            mod_frame: np.ndarray,
            transparency: float = 0.2,
    ) -> np.ndarray:

        shape = mod_frame.shape
        foreground_coloring = np.zeros((3, shape[1]), dtype=np.float32)
        foreground = np.zeros(shape, dtype=np.float32)
        background = np.zeros(shape, dtype=np.float32)
        all_green = np.zeros((3, shape[1]), dtype=np.float32)
        all_green[1] = 255
        classified_pixels = np.apply_along_axis(func1d=self._is_pixel_in_object, axis=0, arr=mod_frame)

        foreground[:, classified_pixels] = mod_frame[:, classified_pixels]
        background[:, ~classified_pixels] = mod_frame[:, ~classified_pixels]
        foreground_coloring[:, classified_pixels] = all_green[:, classified_pixels]
        foreground_coloring = foreground_coloring.T.reshape(origin_frame.shape)

        marked_frame = cv2.addWeighted(foreground_coloring, transparency, origin_frame, 1 - transparency, 0, dtype=cv2.CV_8U)
        self.fg, self.bg = foreground, background
        return marked_frame

    def _get_foreground_background(self, frame: np.ndarray, show: bool = False) -> (np.ndarray, np.ndarray):
        if self.verbose:
            print("Getting foreground and background of first frame...")

        os.makedirs(f"data/images/{self.name}", exist_ok=True)
        fg_path = f"data/images/{self.name}/foreground_ratio{self.resize_ratio}.png"
        bg_path = f"data/images/{self.name}/background_ratio{self.resize_ratio}.png"
        first_frame_foreground = cv2.imread(fg_path)
        first_frame_background = cv2.imread(bg_path)

        if first_frame_foreground is None or first_frame_background is None:

            if self.verbose:
                print("No foreground and background found, running GrabCut...")

            grab_cut = GrabCut(frame, complement_with_white=self.complement_with_white)
            grab_cut.run(show)

            first_frame_foreground, first_frame_background = grab_cut.foreground, grab_cut.background

            cv2.imwrite(fg_path, first_frame_foreground)
            cv2.imwrite(bg_path, first_frame_background)

            if self.verbose:
                print(f"GrabCut done. Saved first frame foreground and background of {self.name} to data/images/")

        return first_frame_foreground, first_frame_background

    def segment(self,
                frames_num: int = 1000,
                verbose: int = 0,
                use_max_pdf: bool = False,
                alpha: float = 10,
                epsilon: float = 0.001,
                use_niw_prior: bool = False,
                run_fit_partial: bool = True,
                iters_of_fit_partial: int = 10,
                close_tol: float = 1e-7,
                redundancy_tol: float = 1e-7,
                ):

        self.verbose = verbose
        models_verbose = verbose > 1
        self.frame_idx = 0
        self.use_max_pdf = use_max_pdf
        self.close_tol = close_tol
        self.redundancy_tol = redundancy_tol

        video = cv2.VideoCapture(self.video_path)
        ret, frame = video.read()
        if not ret:
            raise Exception(f"Failed to read video {self.video_path}, please move video to {self.data_path}/videos/")
        if self.resize_ratio < 1:
            frame = self._resize_frame(frame)
        shape = frame.shape

        if self.verbose:
            print(f"Loaded first frame from {self.video_path} with size {shape}")

        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video.get(cv2.CAP_PROP_FPS)
        output_path = f"data/new_videos/{self.name}.mp4"
        new_video = cv2.VideoWriter(output_path, fourcc=fourcc, fps=choose_fps(fps, frames_num), frameSize=(width, height))

        self.fg, self.bg = self._get_foreground_background(frame)
        new_frame = self._classify_first_frame(frame)
        os.makedirs(f"data/new_frames/{self.name}", exist_ok=True)
        cv2.imwrite(f"data/new_frames/{self.name}/frame{self.frame_idx}.png", new_frame)
        new_video.write(new_frame)

        if self.verbose:
            print("Initializing models prerequisites...")

        self.frame_idx += 1
        self.fg = self._preprocess_frame(self.fg)
        self.bg = self._preprocess_frame(self.bg)
        # Initialize the models

        fg_prior, bg_prior = self._create_priors(use_niw=use_niw_prior)

        if self.verbose:
            print("Initializing models...")
            print("Running fit_init on first frame foreground and background...")

        fg_model = DPMMPython.fit_init(
            data=self.fg, alpha=alpha, prior=fg_prior, verbose=models_verbose, burnout=5, gt=None, epsilon=epsilon
        )
        bg_model = DPMMPython.fit_init(
            data=self.bg, alpha=alpha, prior=bg_prior, verbose=models_verbose, burnout=5, gt=None, epsilon=epsilon
        )

        if self.verbose:
            print("Done.")
            print("Starting segmentation loop on rest of frames...", end="")

        iterable = range(frames_num) if models_verbose else tqdm(range(frames_num))
        for _ in iterable:
            ret, frame = video.read()
            if not ret:
                break
            if self.resize_ratio < 1:
                frame = self._resize_frame(frame)

            if models_verbose:
                print(f"Segmenting frame {self.frame_idx}...")
                print("Creating gaussian mixtures from models...", end="")

            self.fgm = extract_gaussian_data_and_create_mixture(fg_model)
            self.bgm = extract_gaussian_data_and_create_mixture(bg_model)

            if models_verbose:
                print("Done.")
                print("Preprocessing frame for algorithm...", end="")

            mod_frame = self._preprocess_frame(frame)

            if models_verbose:
                print("Done.")
                print("Classifying pixels to foreground and background...", end="")

            new_frame = self._classify_frame(frame, mod_frame)
            cv2.imwrite(f"data/new_frames/{self.name}/frame{self.frame_idx}.png", new_frame)
            new_video.write(new_frame)

            if run_fit_partial:
                if models_verbose:
                    print("Done.")
                    print("Running fit_partial on new foreground and background...", end="")

                fg_model = DPMMPython.fit_partial(
                    model=fg_model, data=self.fg, t=self.frame_idx, iterations=iters_of_fit_partial)
                bg_model = DPMMPython.fit_partial(
                    model=bg_model, data=self.bg, t=self.frame_idx, iterations=iters_of_fit_partial)

            self.frame_idx += 1

            if models_verbose:
                print("Done.")

        if self.verbose:
            print("Segmentation done.")

        video.release()
        new_video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    gm1 = GaussianMixture([[0, 0, 0]], [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    gm2 = GaussianMixture([[0, 0, 0]], [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    pixel = np.array([0, 0, 0])
    vos = VideoObjectSegmentation(name="test")
    vos.fgm = gm1
    vos.bgm = gm2
    print(vos._is_pixel_in_object(pixel))


