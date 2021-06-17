import numpy as np
import cv2
from mySOT.tracker.utils import cos_window, gaussian2d_rolled_labels, fft2, ifft2, extract_hog_feature


class KCF:
    def __init__(self, padding=1.5, features='hog', kernel='gaussian'):
        self.padding = padding
        self._lambda = 1e-4
        self.features = features
        self.kernel = kernel

        if self.features == 'gray' or self.features == 'color':
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1
            self.output_sigma_factor = 0.1
        elif self.features == 'hog':
            self.interp_factor = 0.02
            self.sigma = 0.5
            self.cell_size = 4
            self.output_sigma_factor = 0.1
        else:
            raise NotImplementedError

    # 初始化参数、图像选择框和特征等
    def init(self, first_frame, bbox):
        assert len(first_frame.shape) == 3 and first_frame.shape[2] == 3
        if self.features == 'gray':
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        bbox = np.array(bbox).astype(np.int64)
        x, y, w, h = tuple(bbox)
        self._center = (np.floor(x + w / 2), np.floor(y + h / 2))
        self.w, self.h = w, h
        self.window_size = (int(np.floor(w * (1 + self.padding))) // self.cell_size,
                            int(np.floor(h * (1 + self.padding))) // self.cell_size)
        self._window = cos_window(self.window_size)

        s = np.sqrt(w * h) * self.output_sigma_factor / self.cell_size
        self.yf = fft2(gaussian2d_rolled_labels(self.window_size, s))

        # 截取框选图像
        if self.features == 'gray' or self.features == 'color':
            first_frame = first_frame.astype(np.float32) / 255
            x = self._crop(first_frame, self._center, (w, h))
            x = x - np.mean(x)
        elif self.features == 'hog':
            x = self._crop(first_frame, self._center, (w, h))
            x = cv2.resize(x, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            x = extract_hog_feature(x, cell_size=self.cell_size)
        else:
            raise NotImplementedError

        self.xf = fft2(self._get_windowed(x, self._window))
        self.init_response_center = (0, 0)
        self.alphaf = self._training(self.xf, self.yf)

    # 更新下一帧的目标选择框
    def update(self, current_frame):
        assert len(current_frame.shape) == 3 and current_frame.shape[2] == 3
        if self.features == 'gray':
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if self.features == 'color' or self.features == 'gray':
            current_frame = current_frame.astype(np.float32) / 255
            z = self._crop(current_frame, self._center, (self.w, self.h))
            z = z - np.mean(z)
        elif self.features == 'hog':
            z = self._crop(current_frame, self._center, (self.w, self.h))
            z = cv2.resize(z, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            z = extract_hog_feature(z, cell_size=self.cell_size)
        else:
            raise NotImplementedError

        # 计算特征和相似性
        zf = fft2(self._get_windowed(z, self._window))
        responses = self._detection(self.alphaf, self.xf, zf, kernel=self.kernel)

        # 更新中心点位置
        curr = np.unravel_index(np.argmax(responses, axis=None), responses.shape)

        if curr[0]+1 > self.window_size[1]/2:
            dy = curr[0] - self.window_size[1]
        else:
            dy = curr[0]
        if curr[1]+1 > self.window_size[0]/2:
            dx = curr[1]-self.window_size[0]
        else:
            dx = curr[1]
        dy, dx = dy*self.cell_size, dx*self.cell_size
        x_c, y_c = self._center
        x_c += dx
        y_c += dy
        self._center = (np.floor(x_c), np.floor(y_c))

        # 更新参数
        if self.features == 'color' or self.features == 'gray':
            new_x = self._crop(current_frame, self._center, (self.w, self.h))
        elif self.features == 'hog':
            new_x = self._crop(current_frame, self._center, (self.w, self.h))
            new_x = cv2.resize(new_x, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            new_x = extract_hog_feature(new_x, cell_size=self.cell_size)
        else:
            raise NotImplementedError
        new_xf = fft2(self._get_windowed(new_x, self._window))
        self.alphaf = self.interp_factor * self._training(new_xf, self.yf, kernel=self.kernel) + (1 - self.interp_factor) * self.alphaf
        self.xf = self.interp_factor * new_xf + (1 - self.interp_factor) * self.xf
        return self._center[0] - self.w / 2, self._center[1] - self.h / 2, self.w, self.h

    # 滤波相关计算
    def _kernel_correlation(self, xf, yf, kernel):
        if kernel == 'gaussian':
            N = xf.shape[0] * xf.shape[1]
            xx = (np.dot(xf.flatten().conj().T, xf.flatten())/N)
            yy = (np.dot(yf.flatten().conj().T, yf.flatten())/N)
            xyf = xf * np.conj(yf)
            xy = np.sum(np.real(ifft2(xyf)), axis=2)
            kf = fft2(np.exp(-1 / self.sigma ** 2 * np.clip(xx+yy-2*xy, a_min=0, a_max=None) / np.size(xf)))
        elif kernel == 'linear':
            kf = np.sum(xf*np.conj(yf), axis=2) / np.size(xf)
        else:
            raise NotImplementedError
        return kf

    def _training(self, xf, yf, kernel='gaussian'):
        kf = self._kernel_correlation(xf, xf, kernel)
        alphaf = yf / (kf + self._lambda)
        return alphaf

    def _detection(self, alphaf, xf, zf, kernel='gaussian'):
        kzf = self._kernel_correlation(zf, xf, kernel)
        responses = np.real(ifft2(alphaf * kzf))
        return responses

    # 截取指定大小的图像
    def _crop(self, img, center, target_sz):
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]  # 扩充成三维
        w, h = target_sz
        # 依中心点截取对应宽高的图像
        cropped = cv2.getRectSubPix(img, (int(np.floor((1 + self.padding) * w)), int(np.floor((1 + self.padding) * h))), center)
        return cropped

    def _get_windowed(self, img, cos_window):
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        windowed = cos_window[:, :, None] * img
        return windowed
