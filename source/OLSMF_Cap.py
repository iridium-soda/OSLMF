import warnings
warnings.filterwarnings("ignore")
import sys
import math
import random
from time import time
sys.path.append("..")
from sklearn.svm import SVC
from evaluation.helpers import *
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from onlinelearning.ensemble import *
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from onlinelearning.online_learning import *
from semi_supervised.semiSupervised import *
from sklearn.preprocessing import StandardScaler
from em.online_expectation_maximization import OnlineExpectationMaximization

if __name__ == '__main__':
    # dataset: wpbc; ionosphere; wdbc; australian; credit; wbc; diabetes; dna; german; splice; kr_vs_kp; magic04; a8a; stream
    dataset = "australian"

    #获取超参数
    contribute_error_rate, window_size_denominator, batch_size_denominator, decay_coef_change, decay_choice, shuffle =\
        get_cap_hyperparameter(dataset)
    MASK_NUM = 1
    if dataset == "Stream1":
        X = pd.read_csv("../dataset/MaskData/" + dataset + "/X.txt", sep = " ", header = None)
    else:
        X = pd.read_csv("../dataset/MaskData/" + dataset + "/X_process.txt", sep = " " , header = None)
    if dataset == "a8a":
        X = abs(X)
    Y_label = pd.read_csv("../dataset/DataLabel/" + dataset + "/Y_label.txt", sep = ' ', header = None)

    # ionosphere, diabetes, credit
    if (dataset == "magic04") & (shuffle == True):
        shufflenum = shuffle_dataset_1(Y_label)

    Y_label_masked = random.sample(range(1, Y_label.shape[0]), int(Y_label.shape[0] * 0.5)) # 随机取样了一半的样本
    Y_label_masked.sort() #排序
    Y_label_masked = np.array(Y_label_masked)
    # Y_label_masked实际上是一个索引，用于标记哪些样本被随机屏蔽了

    X_masked = mask_types(X, MASK_NUM, seed = 1)# 这里将X随机屏蔽一部分数据
    X = X.values
    Y_label = Y_label.values

    if (dataset == "magic04") & (shuffle == True):
        X = X[shufflenum]
        Y_label = Y_label[shufflenum]

    all_cont_indices = get_cont_indices(X_masked)#所有连续型变量的索引
    all_ord_indices  = ~all_cont_indices# 所有有序型变量的索引

    n = X_masked.shape[0] #n是样本的数量=690
    feat = X_masked.shape[1] # feat是样本的特征数量
    Y_label = Y_label.flatten() # 这里是原数据,转换为一维数组
    Y_label_masked = Y_label_masked.flatten() # 随机屏蔽的Y值的index,并转换为一维数组

    #setting hyperparameter
    max_iter = batch_size_denominator * 2
    BATCH_SIZE = math.ceil(n / batch_size_denominator)
    WINDOW_SIZE = math.ceil(n / window_size_denominator)
    NUM_ORD_UPDATES = 1
    batch_c = 8

    # start online copula imputation
    # 将混合类型特征映射到潜在正态分布连续空间
    oem = OnlineExpectationMaximization(all_cont_indices, all_ord_indices, window_size=WINDOW_SIZE)
    j = 0
    # 创建和X_masked一样大小的空数组
    X_imp    = np.empty(X_masked.shape)
    Z_imp    = np.empty(X_masked.shape)
    X_masked = np.array(X_masked)# 转换为np数组
    Y_label_fill_x = np.empty(Y_label.shape)
    Y_label_fill_z = np.empty(Y_label.shape)
    Y_label_fill_x_ensemble = np.empty(Y_label.shape)
    Y_label_fill_z_ensemble = np.empty(Y_label.shape)

    #创建SVM分类器
    clf1 = LinearSVC(random_state=0, tol=1e-5)
    clf2 = LinearSVC(random_state=0, tol=1e-5)
    clf_x = LinearSVC(random_state=0, tol=1e-5)
    clf_z = LinearSVC(random_state=0, tol=1e-5)

    print(f"X:{X}")
    print(f"X_mask:{X_masked}")
    print(f"Y_mask:{Y_label_masked}")
    """
    X:[[1.00000000e+00 1.25263158e-01 4.09285714e-01 ... 2.00000000e+00
  5.00000000e-02 1.21200000e-02]
 [0.00000000e+00 1.34135338e-01 2.50000000e-01 ... 2.00000000e+00
  8.00000000e-02 0.00000000e+00]
 [0.00000000e+00 2.38045113e-01 6.25000000e-02 ... 2.00000000e+00
  1.40000000e-01 0.00000000e+00]
 ...
 [0.00000000e+00 7.63909774e-02 3.40714286e-01 ... 2.00000000e+00
  5.00000000e-02 0.00000000e+00]
 [0.00000000e+00 2.05563910e-01 5.17857143e-01 ... 2.00000000e+00
  6.00000000e-02 1.10000000e-04]
 [1.00000000e+00 4.09774436e-01 1.42857143e-03 ... 1.00000000e+00
  2.80000000e-01 0.00000000e+00]]
X_mask:[[       nan 0.12526316        nan ... 2.         0.05              nan]
 [0.                nan 0.25       ... 2.         0.08              nan]
 [       nan 0.23804511        nan ...        nan        nan 0.        ]
 ...
 [0.                nan 0.34071429 ...        nan 0.05              nan]
 [0.                nan        nan ... 2.         0.06              nan]
 [       nan 0.40977444        nan ... 1.                nan 0.        ]]
Y_mask:[  1   2   4   7   8   9  10  11  12  14  16  17  21  25  27  28  30  32
  34  35  37  38  39  40  42  43  46  48  49  54  55  59  62  66  74  80
  81  83  85  86  92  96  97 100 104 106 107 108 109 110 111 112 113 114
 115 117 120 123 125 126 128 131 132 136 141 142 145 146 150 153 154 155
 158 161 162 163 164 165 166 167 168 169 171 173 175 179 181 182 184 190
 194 198 199 204 206 207 208 209 211 214 216 217 220 221 222 223 226 227
 230 232 233 234 235 239 240 241 242 243 244 248 249 252 253 254 255 257
 258 260 261 262 264 268 269 270 271 272 273 274 275 277 278 279 280 285
 287 289 290 293 294 295 297 299 300 304 305 307 308 309 311 313 317 319
 321 323 324 327 330 331 335 336 338 341 344 345 347 349 353 355 360 361
 365 366 367 368 374 375 379 382 383 387 388 389 391 393 394 398 401 402
 405 406 407 413 414 416 418 420 421 422 423 425 428 429 430 431 432 434
 437 438 440 442 443 446 447 448 452 454 456 459 460 462 465 469 470 472
 475 478 479 482 484 485 487 488 493 494 496 498 499 502 504 505 506 507
 508 509 510 511 512 513 514 516 518 519 521 522 523 524 526 533 536 540
 543 544 548 551 553 554 555 557 561 563 564 565 567 569 570 571 575 576
 577 578 579 581 583 587 589 594 597 604 605 606 607 608 610 611 612 614
 616 617 619 621 622 625 626 628 633 634 638 641 642 643 645 646 647 653
 654 656 657 658 659 660 663 664 666 667 669 671 673 674 675 676 677 679
 682 685 689]
 """
    
    #迭代max_iter次
    while j <= max_iter:
        # 选择一段batch
        start = (j * BATCH_SIZE) % n
        end   = ((j + 1) * BATCH_SIZE) % n
        
        #如果end < start,则将end之前的数据和start之后的数据拼接起来
        if end < start:
            indices = np.concatenate((np.arange(end), np.arange(start, n, 1))) # 生成某个范围或区间对应的数组
        else:
            indices = np.arange(start, end, 1)

        # 衰减系数
        if decay_coef_change == 1:
            this_decay_coef = batch_c / (j + batch_c)
        else:
            this_decay_coef = 0.5
        # 使用X_masked的数据进行训练,使用 X_batch 中的数据更新 copula 的拟合度，并返回插补值和 copula 的新相关性
        # 返回值是重建缺失值后的样本
        Z_imp[indices, :], X_imp[indices, :] = oem.partial_fit_and_predict(X_masked[indices, :], max_workers = 1, decay_coef = this_decay_coef)

        #选取训练样本和对应标签
        if start == 0:
            train_x, label_train_x, initial_label_x = X_imp[indices, :], Y_label[indices, ], Y_label_masked[(Y_label_masked > start) & (Y_label_masked < end) ]
            train_z, label_train_z, initial_label_z = Z_imp[indices, :], Y_label[indices, ], Y_label_masked[(Y_label_masked > start) & (Y_label_masked < end) ]
        elif end == 0:
            train_x, label_train_x, initial_label_x = X_imp[indices, :], Y_label[indices, ], Y_label_masked[(Y_label_masked > start) & (Y_label_masked < ((j + 1) * BATCH_SIZE))] % start
            train_z, label_train_z, initial_label_z = Z_imp[indices, :], Y_label[indices, ], Y_label_masked[(Y_label_masked > start) & (Y_label_masked < ((j + 1) * BATCH_SIZE))] % start
        elif end < start:
            initial_label_x = Y_label_masked[(Y_label_masked > start)]
            initial_label_x = initial_label_x.tolist()
            initial_label_x.extend(Y_label_masked[(Y_label_masked < end)])
            initial_label_x = np.array(initial_label_x)
            np.sort(initial_label_x)
            initial_label_x = initial_label_x % len(indices)
            train_x, label_train_x = X_imp[indices, :], Y_label[indices,]

            initial_label_z = Y_label_masked[(Y_label_masked > start)]
            initial_label_z = initial_label_z.tolist()
            initial_label_z.extend(Y_label_masked[(Y_label_masked < end)])
            initial_label_z = np.array(initial_label_z)
            np.sort(initial_label_z)
            initial_label_z = initial_label_z % len(indices)
            train_z, label_train_z = Z_imp[indices, :], Y_label[indices,]
        else:
            train_x, label_train_x, initial_label_x = X_imp[indices, : ], Y_label[indices, ], Y_label_masked[(Y_label_masked > start) & (Y_label_masked < end) ] % start
            train_z, label_train_z, initial_label_z = Z_imp[indices, : ], Y_label[indices, ], Y_label_masked[(Y_label_masked > start) & (Y_label_masked < end) ] % start

        percent = 5
        
        # 计算峰值数据密度并查找指向结构关系的数据
        nneigh_x = DensityPeaks(train_x, percent)
        nneigh_z = DensityPeaks(train_z, percent)

        predict_label_train_x_ensemble = SSC_DensityPeaks_SVC_ensemble(train_x, label_train_x, train_z, label_train_z, initial_label_x, nneigh_x, nneigh_z, clf1, clf2)
        predict_label_train_z_ensemble = SSC_DensityPeaks_SVC_ensemble(train_z, label_train_z, train_x, label_train_x, initial_label_z, nneigh_z, nneigh_x, clf1, clf2)
        Y_label_fill_x_ensemble[indices, ] = predict_label_train_x_ensemble
        Y_label_fill_z_ensemble[indices, ] = predict_label_train_z_ensemble

        predict_label_train_x = SSC_DensityPeaks_SVC(train_x, label_train_x, initial_label_x, nneigh_x, clf_x)
        Y_label_fill_x[indices,] = predict_label_train_x

        predict_label_train_z = SSC_DensityPeaks_SVC(train_z, label_train_z, initial_label_z, nneigh_z, clf_z)
        Y_label_fill_z[indices, ] = predict_label_train_z

        j += 1

    # geting and drawing the error of Semi-Supervised
    Cumulative_error_rate_semi(Y_label_fill_x_ensemble,
                               Y_label_fill_z_ensemble,
                               Y_label_fill_x,
                               Y_label_fill_z,
                               Y_label,
                               dataset)

    temp = np.ones((n, 1))

    X_masked = pd.DataFrame(X_masked)
    X_zeros = X_masked.fillna(value=0)
    X_input_zero = np.hstack((temp, X_zeros))
    if shuffle == True:
        perm = np.arange(n)
        np.random.seed(1)
        np.random.shuffle(perm)
        Y_label = Y_label[perm]
        X_input_zero = X_input_zero[perm]
        Y_label_fill_x = Y_label_fill_x[perm]
        Y_label_fill_z = Y_label_fill_z[perm]

    batch = math.ceil(n / 100)

    X_Zero_CER_fill, svm_error = generate_X_Y(n, X_input_zero, Y_label_fill_x, Y_label, decay_choice, contribute_error_rate)

    X_Zero_CER = generate_Xmask(n, X_input_zero, Y_label, Y_label_masked, decay_choice, contribute_error_rate)

    #get the error of latent space
    temp_zim = np.ones((n, 1))
    X_input_z_imp = np.hstack((temp, Z_imp))
    if shuffle == True:
        perm = np.arange(n)
        np.random.seed(1)
        np.random.shuffle(perm)
        X_input_z_imp = X_input_z_imp[perm]

    Z_impl_CER = generate_Xmask(n, X_input_z_imp, Y_label, Y_label_masked, decay_choice, contribute_error_rate)
    Z_impl_CER_fill, svm_error_z = generate_X_Y(n, X_input_z_imp, Y_label_fill_z, Y_label, decay_choice, contribute_error_rate)

    ensemble_XZ_imp_CER_fill, lamda_array_XZ_fill = ensemble_Y(n, X_input_z_imp, X_input_zero, Y_label, Y_label_fill_x, decay_choice, contribute_error_rate)

    ensemble_XZ_imp_CER = ensemble_Xmask(n, X_input_z_imp, X_input_zero, Y_label, Y_label_masked, decay_choice, contribute_error_rate)

    draw_cap_error_picture(ensemble_XZ_imp_CER_fill,
                           X_Zero_CER_fill,
                           ensemble_XZ_imp_CER,
                           X_Zero_CER,
                           Z_impl_CER,
                           Z_impl_CER_fill,
                           svm_error,
                           dataset)
