import cv2
from scipy.stats import wasserstein_distance
from matplotlib import pyplot as plt
import numpy as np
import skimage
from skimage.metrics import structural_similarity as ssim
from yolox.tracker import matching
import os

def pixel_distribution(img):
    new_img = img
    hist_b = cv2.calcHist([new_img],[0],None,[256],[0,256])       # blue channel
    hist_g = cv2.calcHist([new_img],[1],None,[256],[0,256])       # green channel
    hist_r = cv2.calcHist([new_img],[2],None,[256],[0,256])       # red channel

    cv2.normalize(hist_b,hist_b,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_g,hist_g,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_r,hist_r,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return (hist_b.flatten(), hist_g.flatten(), hist_r.flatten())

def pixel_distribution_mono(img):
    new_img = img
    hist_b = cv2.calcHist([new_img],[0],None,[256],[0,256])       # blue channel

    cv2.normalize(hist_b,hist_b,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return (hist_b.flatten(), hist_b.flatten(), hist_b.flatten())

def distribution_scaling(dis, coef):
    new_dis = np.zeros(256)
    for i in range(256):
        idx_new = int(i * coef)
        if idx_new >= 255:
            idx_new = 255
        new_dis[idx_new] = dis[i]
    return new_dis

def scaling_wasserstein_distance(mean_a, dis_a, new_img, thre):
    hist_b = cv2.calcHist([new_img],[0],None,[256],[0,256])       # blue channel
    hist_g = cv2.calcHist([new_img],[1],None,[256],[0,256])       # green channel
    hist_r = cv2.calcHist([new_img],[2],None,[256],[0,256])       # red channel

    cv2.normalize(hist_b,hist_b,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_g,hist_g,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_r,hist_r,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_b = hist_b.flatten()
    hist_g = hist_g.flatten()
    hist_r = hist_r.flatten()
    # scaling the brightness
    coeff = 1
    print(new_img.mean(), mean_a)
    if coeff >= 1:
        new_img = np.array(new_img/coeff,dtype=np.uint8)
    else:
        new_img = np.array(new_img/coeff,dtype=np.uint8)
        new_img[new_img>255] = 255

    n_b, n_g, n_r = pixel_distribution(new_img)
    o_b, o_g, o_r = dis_a
    
    b_s = wasserstein_distance(n_b, o_b, n_b, n_b)
    g_s = wasserstein_distance(n_g, o_g, n_g, n_g)
    r_s = wasserstein_distance(n_r, o_r, n_r, n_r)
    print(b_s, g_s, r_s)

    if (b_s + g_s + r_s)/3 < thre:
        return True
    else:
        return False 

def dynamic_chunks(img,row,col):
    height, width = img.shape[:2]
    row_interval = int(height/row)
    col_interval = int(width/col)
    quantization_list = np.empty((row,col),dtype=object)
    for i in range(row):
        for j in range(col):
            quantization_list[i][j] = (i*row_interval, j*col_interval, col_interval, row_interval)
    return quantization_list 

def quantization(img,quant_list,row,col):
    height, width = img.shape[:2]
    fig, axes = plt.subplots(row,col)
    image_partition_list = np.empty((row,col),dtype=object)
    for i in range(row):
        for j in range(col):
            y, x, w, h = quant_list[i][j]
            top_x = x
            top_y = y
            bot_x = top_x + w
            bot_y = top_y + h
            if bot_x >= width:
                bot_x = width-1
            if bot_y >= height:
                bot_y = height-1
            img_partition = img[top_y:bot_y, top_x:bot_x]
            image_partition_list[i][j] = img_partition
            cv2.imwrite(os.getcwd()+"/imgs/"+"00_a"+str({i})+str({j})+".png", img_partition)
    return image_partition_list

def my_distribution_plot(quan_a, quan_c, row, col):
    fig, axes = plt.subplots(3,3,figsize=(10, 5))
    ct = 0
    for i in range(row):
        for j in range(col):
            new_img = quan_a[i][j]
            
            hist_b = cv2.calcHist([new_img],[0],None,[256],[0,256])       # blue channel
            hist_g = cv2.calcHist([new_img],[1],None,[256],[0,256])       # green channel
            hist_r = cv2.calcHist([new_img],[2],None,[256],[0,256])       # red channel

            cv2.normalize(hist_b,hist_b,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_g,hist_g,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_r,hist_r,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            hist_b = hist_b.flatten()
            hist_g = hist_g.flatten()
            hist_r = hist_r.flatten()
            if ct <=2:
                print(i)
                print(j)
                axes[i*col+j][0].plot(hist_b, color='b')
                axes[i*col+j][1].plot(hist_g, color='g')
                axes[i*col+j][2].plot(hist_r, color='r')
                ct+=1
    #plt.xlabel("Pixel Intensity")
    plt.savefig("a_quant.png")

    fig, axes = plt.subplots(3,3, figsize=(10, 5))
    ct = 0
    for i in range(row):
        for j in range(col):
            new_img = quan_c[i][j]
            
            hist_b = cv2.calcHist([new_img],[0],None,[256],[0,256])       # blue channel
            hist_g = cv2.calcHist([new_img],[1],None,[256],[0,256])       # green channel
            hist_r = cv2.calcHist([new_img],[2],None,[256],[0,256])       # red channel

            cv2.normalize(hist_b,hist_b,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_g,hist_g,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_r,hist_r,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            hist_b = hist_b.flatten()
            hist_g = hist_g.flatten()
            hist_r = hist_r.flatten()
            if ct <=2:
                axes[i*col+j][0].plot(hist_b, color='b')
                axes[i*col+j][1].plot(hist_g, color='g')
                axes[i*col+j][2].plot(hist_r, color='r')
                ct+=1
    #plt.xlabel("Pixel Intensity")
    plt.savefig("c_quant.png")


def my_wass(quan_a, quan_b, row, col):
    wass_matrix = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            new_img_a = quan_a[i][j]
            
            hist_b_a = cv2.calcHist([new_img_a],[0],None,[256],[0,256])       # blue channel
            hist_g_a = cv2.calcHist([new_img_a],[1],None,[256],[0,256])       # green channel
            hist_r_a = cv2.calcHist([new_img_a],[2],None,[256],[0,256])       # red channel

            cv2.normalize(hist_b_a,hist_b_a,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_g_a,hist_g_a,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_r_a,hist_r_a,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            hist_b_a = hist_b_a.flatten()
            hist_g_a = hist_g_a.flatten()
            hist_r_a = hist_r_a.flatten()

            new_img_b = quan_b[i][j]
            
            hist_b_b = cv2.calcHist([new_img_b],[0],None,[256],[0,256])       # blue channel
            hist_g_b = cv2.calcHist([new_img_b],[1],None,[256],[0,256])       # green channel
            hist_r_b = cv2.calcHist([new_img_b],[2],None,[256],[0,256])       # red channel

            cv2.normalize(hist_b_b,hist_b_b,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_g_b,hist_g_b,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_r_b,hist_r_b,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            hist_b_b = hist_b_b.flatten()
            hist_g_b = hist_g_b.flatten()
            hist_r_b = hist_r_b.flatten()

            b_s = wasserstein_distance(hist_b_a, hist_b_b, hist_b_a, hist_b_a)
            g_s = wasserstein_distance(hist_g_a, hist_g_b, hist_g_a, hist_g_a)
            r_s = wasserstein_distance(hist_r_a, hist_r_b, hist_r_a, hist_r_a)
            was_matrix[i][j] = (b_s + g_s + r_s)/3
    return wass_matrix


def quantization_color_vector(img,quant_list,row,col):
    height, width = img.shape[:2]
    img = img/255
    image_partition_list = np.empty((row,col),dtype=object)
    for i in range(row):
        for j in range(col):
            y, x, w, h = quant_list[i][j]
            top_x = x
            top_y = y
            bot_x = top_x + w
            bot_y = top_y + h
            if bot_x >= width:
                bot_x = width-1
            if bot_y >= height:
                bot_y = height-1
            img_partition = img[top_y:bot_y, top_x:bot_x]
            image_partition_list[i][j] = img_partition
    return image_partition_list


def color_vector_calculation(quan_a, quan_b, row, col):
    cost_matrix = np.zeros((row*col, row*col))

    for i_a in range(row):
        for j_a in range(col):
            
            new_img_a = quan_a[i_a][j_a]
            pix_a = new_img_a.shape[0]*new_img_a.shape[1]
                    
            suma_b = np.sum(new_img_a[:,:,0])/pix_a      # blue channel
            suma_g = np.sum(new_img_a[:,:,1])/pix_a      # green channel
            suma_r = np.sum(new_img_a[:,:,2])/pix_a      # red channel
            
            a_color = np.array([suma_b, suma_g, suma_r])

            for i_b in range(row):
                for j_b in range(col):

                    new_img_b = quan_b[i_b][j_b]
                    pix_b = new_img_b.shape[0]*new_img_b.shape[1]
                    
                    sumb_b = np.sum(new_img_b[:,:,0])/pix_b      # blue channel
                    sumb_g = np.sum(new_img_b[:,:,1])/pix_b      # green channel
                    sumb_r = np.sum(new_img_b[:,:,2])/pix_b      # red channel
                    # print(sumb_b)
                    # print(sumb_g)
                    # print(sumb_r)
                    if sumb_b ==0 and sumb_g ==0 and sumb_r ==0:
                        print(new_img_b)
                        import sys
                        sys.exit()

                    b_color =  np.array([sumb_b, sumb_g, sumb_r])

                    dis = 1 - np.dot(a_color, b_color)/(np.linalg.norm(a_color)*np.linalg.norm(b_color))

                    cost_matrix[i_a*col+j_a][i_b*col+j_b] = dis
                   
    return cost_matrix
        

if __name__=="__main__":

    a = cv2.imread("/home/dcsl/Documents/Video_Colab/imgs/7_2.png")
   # b = cv2.imread("/home/dcsl/Documents/Video_Colab/imgs/7_2.png")
   # c = cv2.imread("/home/dcsl/Documents/Video_Colab/img_pairs/tt1.png")


    a_mean = a.mean()
   # b_mean = b.mean()

    quan_a = dynamic_chunks(a,6,3)
    quan_a = quantization(a,quan_a,6,3)

  #  quan_b = dynamic_chunks(b,6,3)
  #  quan_b = quantization(b,quan_b,6,3)

  #  my_distribution_plot(quan_a, quan_b, 7, 3)

    # quan_b = dynamic_chunks(b,3,3)
    # quan_b = quantization(b,quan_b,3,3)

   # my_distribution_plot(quan_a, quan_b, 3, 2)
    # lista = quantization_color_vector(a,quan_a,3,2)

    # quan_b = dynamic_chunks(b,3,2)
    # listb = quantization_color_vector(b,quan_b,3,2)

    # quan_c = dynamic_chunks(c,4,3)
    # listc = quantization_color_vector(c,quan_c,4,3)

    # dis = color_vector_calculation(lista, listb, 4,3)
    # matches, u_track, u_detection = matching.linear_assignment(dis, thresh=0.0001)
    # print(np.sum(dis< 0.001))
    # print(matches.shape)

    # dis = color_vector_calculation(lista, listc, 4,3)
    # matches, u_track, u_detection = matching.linear_assignment(dis, thresh=0.0001)
    # print(np.sum(dis< 0.001))
    # print(matches)

    



#     t_b, t_g, t_r = pixel_distribution(a)
#     d_b, d_g, d_r = pixel_distribution_mono(b)
#    # l_b, l_g, l_r = pixel_distribution_mono(c)
    


#     b_s = wasserstein_distance(t_b, d_b, t_b, t_b)
#     g_s = wasserstein_distance(t_g, d_g, t_g, t_g)
#     r_s = wasserstein_distance(t_r, d_r, t_r, t_r)

#     b_l = wasserstein_distance(l_b, d_b, l_b, l_b)
#     g_l = wasserstein_distance(l_g, d_g, l_g, l_g)
#     r_l = wasserstein_distance(l_r, d_r, l_r, l_r)


    # fig, axes = plt.subplots(2,4)

    # axes[0,0].imshow(a)
    # axes[0,0].set_title('last tracked image')

    # axes[0,1].plot(t_r, color='r')
    # axes[0,1].set_title('R')

    # axes[0,2].plot(t_g, color='g')
    # axes[0,2].set_title('G')

    # axes[0,3].plot(t_b, color='b')
    # axes[0,3].set_title('B')

    # axes[1,0].imshow(b)
    # axes[1,0].set_title('detected image')

    # axes[1,1].plot(d_r, color='r')
    # axes[1,1].set_title('R')

    # axes[1,2].plot(d_g, color='g')
    # axes[1,2].set_title('G')

    # axes[1,3].plot(d_b, color='b')
    # axes[1,3].set_title('B')
    # plt.tight_layout()
    # plt.savefig("compare_hist_dynamic_weight.png")


    # fig, axes = plt.subplots(2,4)
    # axes[0,0].imshow(c)
    # axes[0,0].set_title('unscaled image')

    # axes[0,1].plot(l_r, color='r')
    # axes[0,1].set_title('R')

    # axes[0,2].plot(l_g, color='g')
    # axes[0,2].set_title('G')

    # axes[0,3].plot(l_b, color='b')
    # axes[0,3].set_title('B')

    # axes[1,0].imshow(b)
    # axes[1,0].set_title('detected image')

    # axes[1,1].plot(d_r, color='r')
    # axes[1,1].set_title('R')

    # axes[1,2].plot(d_g, color='g')
    # axes[1,2].set_title('G')

    # axes[1,3].plot(d_b, color='b')
    # axes[1,3].set_title('B')
    # plt.tight_layout()
    # plt.savefig("compare_hist_light_unscaled.png")

    
