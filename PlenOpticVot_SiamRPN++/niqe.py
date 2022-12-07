import numpy as np
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
import scipy
from PIL import Image
import scipy.ndimage
import scipy.special
import math
import matplotlib.pyplot as plt

gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
prec_gammas = a/(b*c)

def aggd_features(imdata):
    #flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata*imdata
    left_data = imdata2[imdata<0]
    right_data = imdata2[imdata>=0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
      gamma_hat = left_mean_sqrt/right_mean_sqrt
    else:
      gamma_hat = np.inf
    #solve r-hat norm

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
      r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata2))
    else:
      r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1)*(gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    #solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm)**2);
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0/alpha)
    gam2 = scipy.special.gamma(2.0/alpha)
    gam3 = scipy.special.gamma(3.0/alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    #mean parameter
    N = (br - bl)*(gam2 / gam1)#*aggdratio
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

def ggd_features(imdata):
    nr_gam = 1/prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq/E**2
    pos = np.argmin(np.abs(nr_gam - rho));
    return gamma_range[pos], sigma_sq

def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights

def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
      avg_window = gen_gauss_window(3, 7.0/6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    return (image - mu_image)/(var_image + C), var_image, mu_image


def _niqe_extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl+br)/2.0,
            alpha1, N1, bl1, br1,  # (V)
            alpha2, N2, bl2, br2,  # (H)
            alpha3, N3, bl3, bl3,  # (D1)
            alpha4, N4, bl4, bl4,  # (D2)
    ])

def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)

def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)

def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = np.int(patch_size)
    patches = []
    for j in range(0, h-patch_size+1, patch_size):
        for i in range(0, w-patch_size+1, patch_size):
            patch = img[j:j+patch_size, i:i+patch_size]
            patches.append(patch)

    patches = np.array(patches)
    
    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features

def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % patch_size)
    woffset = (w % patch_size)

    if hoffset > 0: 
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]


    img = img.astype(np.float32)
    # img2 = scipy.misc.imresize(img, 0.5, interp='bicubic', mode='F')
    # img2 = np.array(Image.fromarray(img).resize(h/2, w/2), PIL.Image.BICUBIC).astype(np.double)
    img2 = np.array(Image.fromarray(img).resize((int(h / 2), int(w / 2)), resample=3)).astype(np.double)

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)


    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size/2)

    feats = np.hstack((feats_lvl1, feats_lvl2))# feats_lvl3))

    return feats

def niqe(inputImgData):

    patch_size = 96
    module_path = dirname(__file__)

    # TODO: memoize
    params = scipy.io.loadmat(join(module_path, 'niqe_param', 'niqe_image_params (1).mat'))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]


    M, N = inputImgData.shape

    # assert C == 1, "niqe called with videos containing %d channels. Please supply only the luminance channel" % (C,)
    assert M > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert N > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"


    feats = get_patches_test_features(inputImgData, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = ((pop_cov+sample_cov)/2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))

    return niqe_score

def sharpness1(arr2d):
    gy, gx = np.gradient(arr2d)
    gnorm = np.sqrt(gx ** 2 + gy ** 2)
    sharpness = np.average(gnorm)
    return sharpness

def sharpness2(arr2d):
    dx = np.diff(arr2d)[1:, :]  # remove the first row
    dy = np.diff(arr2d, axis=0)[:, 1:]  # remove the first column
    dnorm = np.sqrt(dx ** 2 + dy ** 2)
    sharpness = np.average(dnorm)
    return sharpness

def sharpness_index(frame_index, focal_range, center, count):
    sh1 = []
    focal_list = []
    for i in range(focal_range[0], focal_range[1]+1):
        path = f"D:/dataset/NonVideo3/{frame_index:0=3d}/focal/{i:0=3d}.png"  # 077 - center(828, 384)
        x, y = (int(center[0]), int(center[1]))
        img = Image.open(path)
        w, h = img.width, img.height
        d = 128 + 20

        # Tracker의 중심점을 기준으로 (255, 255) 영역으로 선명도 계산
        if x - d < 0:
            x_range = (0, 256 + 20)
        elif x + d > w:
            x_range = (w-(256+20), w)
        else:
            x_range = (x-d, x+d)
        if y-d < 0:
            y_range = (0, 256+20)
        elif y+d > h:
            y_range = (h-(256+20), h)
        else:
            y_range = (y-d, y+d)
        img_c = np.array(img.convert('LA'))[y_range[0]:y_range[1]+1, x_range[0]:x_range[1]+1, 0]
        # img_c = np.array(Image.open(path).convert('LA'))[center[1]-(128+20): center[1]+(129+20), center[0]-(128+20): center[0]+(129+20), 0]
        sh1_value = sharpness1(img_c)  # f2, 점수가 높을수록 선명도 높음
        sh1.append((i, sh1_value))

        # print('{}th NIQE: {}, sh1: {}, sh2: {}'.format(i, q_value, sh1_value, sh2_value))
    # print(f'max value : NIQE-[{max(q)}] sh1-[{max(sh1)})] sh2-[{max(sh2)}]')

    sh1_list = sorted(sh1, key=lambda x: x[1], reverse=True)

    # first
    # index_list = sorted(sh1_list[0:count], key=lambda x: x[0])
    # for i in range(index_list[0][0]-3, index_list[count-1][0]+4):
    #     if focal_range[0] <= i <= focal_range[1]:
    #         focal_list.append(i-focal_range[0])
    # focal_list = list(focal_list)

    for d, (k, _) in enumerate(sh1_list[:count]):
        d = count-d
        # d = count
        for i in range(k-d, k+d+1):
            if focal_range[0] <= i <= focal_range[1]:
                focal_list.append(i-focal_range[0])
    focal_list = set(focal_list)
    focal_list = list(focal_list)
    # print(focal_list)
    return focal_list




if __name__ == "__main__":
    sh1 = []
    sh2 = []
    q = []
    for i in range(100):
        # path = "D:/dataset/NonVideo3/077/focal/{0:0=3d}.png".format(i)  # 077 - center(828, 384)
        # path = "D:/dataset/NonVideo3/100/focal/{0:0=3d}.png".format(i)  # 077 - center(828, 384)
        path = f"D:/dataset/NonVideo3/100/focal/{i:0=3d}.png"
        center = (915, 385)
        # img = np.array(Image.open(path).convert('LA'))[:, :, 0]
        img = Image.open(path)

        img = np.array(Image.open(path).convert('LA'))[center[1]-128:center[1]+129, center[0]-128:center[0]+129, 0]
        q_value = niqe(img)  # f1, 점수가 낮을수록 선명도 높음.
        sh1_value = sharpness1(img)  # f2, 점수가 높을수록 선명도 높음
        sh2_value = sharpness2(img)  # f3, 점수가 높을수록 선명도 높음
        q.append((i, q_value))
        sh1.append((i, sh1_value))
        sh2.append((i, sh2_value))
        print('{}th NIQE: {}, sh1: {}, sh2: {}'.format(i, q_value, sh1_value, sh2_value))
    # print(f'max value : NIQE-[{max(q)}] sh1-[{max(sh1)})] sh2-[{max(sh2)}]')
    q_list = sorted(q, key=lambda x: x[1])
    sh1_list = sorted(sh1, key=lambda x: x[1], reverse=True)
    sh2_list = sorted(sh2, key=lambda x: x[1], reverse=True)

    print(q_list[:5])
    print(sh1_list[:5])
    print(sh2_list[:5])

    # img
    for i, (index, _) in enumerate(q_list[:5]):
        path = "D:/dataset/NonVideo3/100/focal/{0:0=3d}.png".format(index)
        img = np.array(Image.open(path))[center[1]-128:center[1]+129, center[0]-128:center[0]+129, :]
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
    plt.show()

    for i, (index, _) in enumerate(sh1_list[:5]):
        path = "D:/dataset/NonVideo3/100/focal/{0:0=3d}.png".format(index)
        img = np.array(Image.open(path))[center[1]-128:center[1]+129, center[0]-128:center[0]+129, :]
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
    plt.show()

    for i, (index, _) in enumerate(sh2_list[:5]):
        path = "D:/dataset/NonVideo3/100/focal/{0:0=3d}.png".format(index)
        img = np.array(Image.open(path))[center[1]-128:center[1]+129, center[0]-128:center[0]+129, :]
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
    plt.show()

    # print(f'max value : NIQE-[{q.index(max(q))}] sh1-[{sh1.index(max(sh1))})] sh2-[{sh2.index(max(sh2))}]')
        # print('{}th sh1: {}, sh2: {}'.format(i, sh1_value, sh2_value))

        # fig, ax = plt.subplots(nrows=2, ncols=2)
        # ax[0, 0].imshow(img, vmin=0, vmax=255)
        # ax[0, 0].set_title("{0:0=3d}th focal plane".format(i))
        #
        # ax[0, 1].bar([0, 1, 2], [0, q_value, 0])
        # ax[0, 1].axis(ymin=0, ymax=20)
        # ax[0, 1].grid()
        # ax[0, 1].set_title("NIQE: {:.6f}".format(q_value))
        #
        # ax[1, 0].bar([0, 1, 2], [0, sh1_value, 0])
        # ax[1, 0].axis(ymin=0, ymax=3)
        # ax[1, 0].grid()
        # ax[1, 0].set_title("Sharpness1: {:.6f}".format(sh1_value))
        #
        # ax[1, 1].bar([0, 1, 2], [0, sh2_value, 0])
        # ax[1, 1].axis(ymin=0, ymax=3)
        # ax[1, 1].grid()
        # ax[1, 1].set_title("Sharpness2: {:.6f}".format(sh2_value))
        #
        # plt.tight_layout()
        # plt.savefig("./focal_{0:0=3d}.png".format(i), dpi=300)
        # plt.close()