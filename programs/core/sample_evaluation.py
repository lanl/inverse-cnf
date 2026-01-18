from programs.utils.logger_setup import get_logger
from programs.utils.common import FLOAT_EPS, List, Tuple, Dict, np, combinations, convert_type, validate_image_pair
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

class ImageContinuousMetrics:

    @staticmethod
    def _mse(real_img: np.ndarray, pred_img: np.ndarray, **kwargs) -> float:
        """Compute Mean Squared Error (MSE) between two images or scalars."""
        if np.isscalar(real_img) and np.isscalar(pred_img):
            return (real_img - pred_img) ** 2  # MSE for scalars
        return np.mean((real_img - pred_img) ** 2)  # Standard MSE for images

    @staticmethod
    def _rmse(real_img: np.ndarray, pred_img: np.ndarray, **kwargs) -> float:
        """Compute Root Mean Squared Error (RMSE) between two images or scalars."""
        return np.sqrt(ImageContinuousMetrics._mse(real_img, pred_img))

    @staticmethod
    def _mae(real_img: np.ndarray, pred_img: np.ndarray, **kwargs) -> float:
        """Compute Mean Absolute Error (MAE) between two images or scalars."""
        if np.isscalar(real_img) and np.isscalar(pred_img):
            return np.abs(real_img - pred_img)  # MAE for scalars
        return np.mean(np.abs(real_img - pred_img))  # MAE for images

    @staticmethod
    def _mape(real_img: np.ndarray, pred_img: np.ndarray, **kwargs) -> float:
        """Compute Mean Absolute Percentage Error (MAPE) between two images or scalars."""
        if np.isscalar(real_img) and np.isscalar(pred_img):
            return np.abs((real_img - pred_img) / np.maximum(real_img, FLOAT_EPS)) * 100
        percent_error = np.abs((real_img - pred_img) / np.maximum(real_img, FLOAT_EPS)) * 100
        return np.mean(percent_error)

    @staticmethod
    def _psnr(real_img: np.ndarray, pred_img: np.ndarray, **kwargs) -> float:
        """Compute Peak Signal Noise Ratio (PSNR) between two images."""
        mse_value = ImageContinuousMetrics._mse(real_img, pred_img)
        if mse_value == 0:
            get_logger().warning("PSNR is undefined for identical images. Returning NaN.")
            return np.nan  # PSNR is undefined for identical images
        max_pixel = max(np.nanmax(real_img), np.nanmax(pred_img))
        return 20 * np.log10(max_pixel / np.sqrt(mse_value))

    @staticmethod
    def _cosine_similarity(real_img: np.ndarray, pred_img: np.ndarray, **kwargs) -> float:
        """Compute Cosine Similarity (CS) between two images or scalars."""
        if np.isscalar(real_img) and np.isscalar(pred_img):
            return 1.0 if real_img == pred_img else 0.0  # Cosine similarity for scalars
        
        # For arrays, flatten and calculate cosine similarity
        real_flat = real_img.flatten().reshape(1, -1) if isinstance(real_img, np.ndarray) else real_img
        pred_flat = pred_img.flatten().reshape(1, -1) if isinstance(pred_img, np.ndarray) else pred_img
        
        return cosine_similarity(real_flat, pred_flat)[0, 0]

    @staticmethod
    def _ssim(
        real_img: np.ndarray,
        pred_img: np.ndarray,
        window_size: int = None,
        data_range: float = None,
        **kwargs
    ) -> float:
        """Compute Structural Similarity Index (SSIM) between two images."""
        if np.isscalar(real_img) and np.isscalar(pred_img):
            return np.nan # SSIM not possible for scalars
        if real_img.shape[-1] < 7:
            return np.nan # SSIM not possible for images smaller than (7,7)
        
        # Handle dimensionality of the images
        if real_img.ndim == 3 and real_img.shape[0] == 1:
            real_img = np.squeeze(real_img)
        if pred_img.ndim == 3 and pred_img.shape[0] == 1:
            pred_img = np.squeeze(pred_img)

        if data_range is None:
            data_range = real_img.max() - real_img.min()

        # Assume the images are grayscale if no channel_axis is specified
        channel_axis = 0 if pred_img.ndim == 3 else None

        return ssim(
            real_img,
            pred_img,
            win_size=window_size,
            data_range=data_range,
            channel_axis=channel_axis,
            multichannel=(channel_axis is not None)
        )

    mse = validate_image_pair(_mse)
    rmse = validate_image_pair(_rmse)
    mae = validate_image_pair(_mae)
    mape = validate_image_pair(_mape)
    psnr = validate_image_pair(_psnr)
    cosine_similarity = validate_image_pair(_cosine_similarity)
    ssim = validate_image_pair(_ssim)

    @staticmethod
    def get_metric_keys() -> list[str]:
        return ["mse", "rmse", "mae", "mape", "psnr", "cos_sim", "ssim"]
    
    @staticmethod
    def get_metric_bounds(metric: str | None = None) -> dict[str, tuple[float, float]] | tuple[float, float]:
        metric_bounds = {
            "mse": (0, None),
            "rmse": (0, None),
            "mae": (0, None),
            "mape": (0, None),
            "psnr": (0, None),
            "cos_sim": (-1, 1),
            "ssim": (0, 1)
        }
        if metric is None or metric == "all":
            return metric_bounds
        elif metric in metric_bounds.keys():
            return metric_bounds[metric]
        return None

    @staticmethod
    @validate_image_pair
    def all_metrics(real_img: np.ndarray, pred_img: np.ndarray, data_range=None) -> dict[str, float]:
        return {
            "mse": ImageContinuousMetrics._mse(real_img, pred_img),
            "rmse": ImageContinuousMetrics._rmse(real_img, pred_img),
            "mae": ImageContinuousMetrics._mae(real_img, pred_img),
            "mape": ImageContinuousMetrics._mape(real_img, pred_img),
            "psnr": ImageContinuousMetrics._psnr(real_img, pred_img),
            "cos_sim": ImageContinuousMetrics._cosine_similarity(real_img, pred_img),
            "ssim": ImageContinuousMetrics._ssim(real_img, pred_img, data_range=data_range),
        }
    
    
class ImageSimilarityComparison:
    """
    Class to compute similarity comparisons between real and predicted images.
    """

    @staticmethod
    def _signed_difference(real_img: np.ndarray, pred_img: np.ndarray) -> np.ndarray:
        """Compute the signed difference map (real - pred)."""
        return real_img - pred_img

    @staticmethod
    def _absolute_difference(real_img: np.ndarray, pred_img: np.ndarray) -> np.ndarray:
        """Compute the absolute difference map."""
        return np.abs(real_img - pred_img)

    def _ssim(real_img: np.ndarray, pred_img: np.ndarray, window_size: int|None = None, gradient:bool=True, full:bool=True) -> np.ndarray:
        window_size = real_img.shape[-1] 

        if real_img.ndim == 3:
            if real_img.shape[0] == 1:
                np.squeeze(real_img)

        if pred_img.ndim == 3:
            if pred_img.shape[0] == 1:
                np.squeeze(pred_img)
        channel_axis = 0 if pred_img.ndim == 3 else None
        
        return ssim(
                    real_img,
                    pred_img,
                    gradiant=gradient,
                    full=full,
                    win_size=window_size,
                    data_range=real_img.max()-real_img.min(),
                    channel_axis=channel_axis,
                    multichannel=(channel_axis is not None)
                )
        

    @staticmethod
    def _ssim_grad(real_img, pred_img):
        _, grad = ImageSimilarityComparison._ssim(real_img, pred_img, full=False)
        return grad
    
    @staticmethod
    def _ssim_image(real_img, pred_img):
        _, image = ImageSimilarityComparison._ssim(real_img, pred_img, gradient=False)
        return image

    def _ssim_all(real_img, pred_img):
        _, grad, image = ImageSimilarityComparison._ssim(real_img, pred_img)
        return grad, image

    signed_difference = validate_image_pair(_signed_difference)
    absolute_difference = validate_image_pair(_absolute_difference)
    ssim_grad = validate_image_pair(_ssim_grad)
    ssim_image = validate_image_pair(_ssim_image)

    @staticmethod
    @validate_image_pair
    def all_comparisons(real_img: np.ndarray, pred_img: np.ndarray) -> Dict[str, np.ndarray]:

        return {
            "signed_diff_image": ImageSimilarityComparison._signed_difference(real_img, pred_img),
            "absolute_diff_image": ImageSimilarityComparison._absolute_difference(real_img, pred_img),
            "overall_ssim_image": ImageSimilarityComparison._ssim_image(real_img, pred_img)
        }

class LatentStatistics:
    """
    Class to evaluate similarity between generated and real images (e.g., φ or εᵣ maps).
    """

    @staticmethod
    def z_mean(z: np.ndarray) -> float:
        return convert_type(np.mean(z), float)
    @staticmethod
    def z_std(z: np.ndarray) -> float:
        return convert_type(np.std(z), float)
    
    @staticmethod
    def z_norm(z: np.ndarray) -> float:
        # Flatten each sample, then compute norm across features
        flat_z = z.reshape(z.shape[0], -1)
        return convert_type(np.mean(np.linalg.norm(flat_z, axis=1)), float)

    @staticmethod
    def kl_divergence(z: np.ndarray) -> float:
        """
        KL divergence between the empirical latent distribution and N(0, I).
        Based on: D_KL(q(z) || p(z)) = 0.5 * sum(z^2 + log(2π) + 1)
        """
        return 0.5 * convert_type(np.mean(z**2 + np.log(2 * np.pi) + 1), float)

    @staticmethod
    def all_stats(z: np.ndarray) -> dict:
        return {
            'z_mean': LatentStatistics.z_mean(z),
            'z_std': LatentStatistics.z_std(z),
            'z_norm': LatentStatistics.z_norm(z),
            'kl_divergence': LatentStatistics.kl_divergence(z)
        }

class ImagePairwiseComparison:

    def __init__(self, images: List[np.ndarray], random_state: int = 1234, tsne_components: int = 2):
        if not images:
            raise ValueError("The images list cannot be empty.")
        self.images: List[np.ndarray] = [img.astype(np.float32) for img in images]
        self.num_samples: int = len(images)
        self.n_components: int = tsne_components
        self.random_state: int = random_state

    def _pairwise_ssim(self) -> np.ndarray:
        """Compute pairwise SSIM for all image pairs."""
        ssim_matrix = np.eye(self.num_samples)
        for i, j in combinations(range(self.num_samples), 2):
            try:
                ssim_val = ssim(
                    self.images[i],
                    self.images[j],
                    win_size=None, 
                    data_range=self.images[i].max() - self.images[i].min(),
                    channel_axis=0,
                    multichannel=True
                )
            except ValueError as e:
                raise ValueError(f"SSIM computation failed for images {i} and {j}: {e}")
            
            ssim_matrix[i, j] = ssim_val
            ssim_matrix[j, i] = ssim_val  # Symmetric
        return ssim_matrix

    def _pairwise_signed_diff(self) -> np.ndarray:
        """Compute pairwise signed differences (mean signed pixel-wise difference)."""
        signed_diff_matrix = np.zeros((self.num_samples, self.num_samples))
        for i, j in combinations(range(self.num_samples), 2):
            signed_diff = np.mean(self.images[i] - self.images[j])
            signed_diff_matrix[i, j] = signed_diff
            signed_diff_matrix[j, i] = -signed_diff  # Reflective
        return signed_diff_matrix

    def _pairwise_abs_diff(self) -> np.ndarray:
        """Compute pairwise absolute differences (mean absolute pixel-wise difference)."""
        abs_diff_matrix = np.zeros((self.num_samples, self.num_samples))
        for i, j in combinations(range(self.num_samples), 2):
            abs_diff = np.mean(np.abs(self.images[i] - self.images[j]))
            abs_diff_matrix[i, j] = abs_diff
            abs_diff_matrix[j, i] = abs_diff  # Symmetric
        return abs_diff_matrix

    def _tsne_coordinates(self) -> np.ndarray:
        """Compute t-SNE coordinates for all images."""
        flattened_images = np.array([img.flatten() for img in self.images])
        tsne = TSNE(n_components=self.n_components, random_state=self.random_state)
        tsne_coordinates = tsne.fit_transform(flattened_images)
        return tsne_coordinates


    def compare_all(self) -> Dict[str, np.ndarray]:
        return {
            "pairwise_ssim": self._pairwise_ssim(),
            "pairwise_signed_diff": self._pairwise_signed_diff(),
            "pairwise_abs_diff": self._pairwise_abs_diff(),
            "tsne_coordinates": self._tsne_coordinates(),
        }


if __name__ == "__main__":
    img1 = np.random.rand(100, 100)
    img2 = np.random.rand(100, 100)
    img3 = img1 + np.random.normal(0, 0.1, (100, 100)) 

    # Compute continuous metrics
    metrics = ImageContinuousMetrics.all_metrics(img1, img2)
    print("Continuous Metrics between img1 and img2:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Compute similarity comparisons
    comparisons = ImageSimilarityComparison.all_comparisons(img1, img2)
    print("\nSimilarity Comparisons between img1 and img2:")
    for comp, array in comparisons.items():
        print(f"{comp}: Mean Value = {array.mean()}")

    # Perform pairwise comparisons for multiple images
    images = [img1, img2, img3]
    pairwise_comp = ImagePairwiseComparison(images)
    comparison_results = pairwise_comp.compare_all()

    print("\nPairwise SSIM Matrix:")
    print(comparison_results["pairwise_ssim"])

    print("\nPairwise Signed Difference Matrix:")
    print(comparison_results["pairwise_signed_diff"])

    print("\nPairwise Absolute Difference Matrix:")
    print(comparison_results["pairwise_abs_diff"])

    print("\nt-SNE Coordinates:")
    print(comparison_results["tsne_coordinates"])
