import openslide
import numpy as np
import os
import datetime
from PIL import Image
from PIL import ImageEnhance


#获取标签
def getLabel(svs_name):
    link_file = open("./label.txt", "r", encoding='UTF-8')
    lines = list(link_file)
    for i in range(len(lines)):
        if (lines[i].split('/')[-1]).split('.')[0] == svs_name:
            label = str(lines[i].split('/')[1])
            break
    link_file.close()
    return label


# 判断patch是否符合
def judge_save(region, new_path, svs_name, patch_size, a):

    image_rgb = region.convert('RGB')
    image_gray = region.convert('L')
    image = np.array(image_gray)

    # 像素值在200及以上为图像中背景部分
    count = int(np.sum(image < 200))

    count_0 = int(np.sum(image < 30))
    if (count / (patch_size * patch_size)) > 0.5 and (count_0/(patch_size * patch_size)) < 0.3:
        # image_rgb.save(new_path + svs_name + '_' + '0' * (digit_num_all - digit_num_i) + str(a + 1) + '.tif')
        # image_rgb.save(new_path + svs_name + '_' + '0' * (digit_num_all - digit_num_i) + str(a + 1) + '.png')
        image_rgb.save(new_path + svs_name + '_' + str(a + 1) + '.jpg')
        return True
    else:
        return False


def deal_patches(slide, patch_size, patch_path, svs_name, k):
    svs_label = getLabel(svs_name)
    whole_path = patch_path + svs_name + '-' + svs_label + '/'
    new_path = whole_path + svs_name + '+' + k + '-' + svs_label + '/'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    x_point = 0
    y_point = 0
    [w, h] = slide.size
    i = 0
    while y_point <= h - patch_size:
        while x_point <= w - patch_size:
            region = slide.crop((x_point, y_point, x_point + patch_size, y_point + patch_size))
            if judge_save(region, new_path, svs_name, patch_size, i):
                i += 1
            x_point += patch_size
        y_point += patch_size
        x_point = 0
    patches_num = i
    return patches_num


def get_patches(img_path, patch_path, patch_size):
    svs_list = [os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.svs')]

    for i in range(len(svs_list)):
        start_time = datetime.datetime.now()
        svs_path = str(svs_list[i])
        svs_name = (svs_path.split('/')[-1]).split('.')[0]
        slide = openslide.open_slide(svs_path)
        print('Deal with ', str(i + 1), 'st images named ', svs_name, end=' ')
        level = 1
        # print(slide.level_count)
        # print(slide.level_downsamples)
        # print(slide.level_dimensions)
        img = slide.read_region((0, 0), level, slide.level_dimensions[level])

        #A~E
        img_rot90 = img.transpose(Image.ROTATE_90)  # 旋转90.
        rot = img.rotate(45)
        fff = Image.new('RGBA', rot.size, (255,) * 4)
        img_rot45 = Image.composite(rot, fff, rot)  # 旋转45.
        img_lr = img.transpose(Image.FLIP_LEFT_RIGHT)  # 左右翻转
        img_tb = img.transpose(Image.FLIP_TOP_BOTTOM)  # 上下翻转

        finished_patches = deal_patches(img, patch_size, patch_path, svs_name, 'A')
        print(',k=A: ', finished_patches, end=' ')
        finished_patches = deal_patches(img_rot90, patch_size, patch_path, svs_name, 'B')
        print(',k=B: ', finished_patches, end=' ')
        finished_patches = deal_patches(img_rot45, patch_size, patch_path, svs_name, 'C')
        print(',k=C: ', finished_patches, end=' ')
        finished_patches = deal_patches(img_lr, patch_size, patch_path, svs_name, 'D')
        print(',k=D: ', finished_patches, end=' ')
        finished_patches = deal_patches(img_tb, patch_size, patch_path, svs_name, 'E')
        print(',k=E: ', finished_patches)

        # F~R
        img_rot180 = img.transpose(Image.ROTATE_180) #F
        img_rot270 = img.transpose(Image.ROTATE_270) #G
        rot = img.rotate(135)
        fff = Image.new('RGBA', rot.size, (255,) * 4)
        img_rot135 = Image.composite(rot, fff, rot)  #H旋转135.
        rot = img.rotate(225)
        fff = Image.new('RGBA', rot.size, (255,) * 4)
        img_rot225 = Image.composite(rot, fff, rot)  #I旋转225.
        rot = img.rotate(315)
        fff = Image.new('RGBA', rot.size, (255,) * 4)
        img_rot315 = Image.composite(rot, fff, rot)  #J旋转315.
        img_bri_h = ImageEnhance.Brightness(img).enhance(factor = 1.3)
        img_bri_l = ImageEnhance.Brightness(img).enhance(factor = 0.7)
        img_col_h = ImageEnhance.Color(img).enhance(factor = 1.4)
        img_col_l = ImageEnhance.Color(img).enhance(factor = 0.6)
        img_con_h = ImageEnhance.Contrast(img).enhance(factor = 1.3)
        img_con_l = ImageEnhance.Contrast(img).enhance(factor = 0.7)
        img_sha_h = ImageEnhance.Sharpness(img).enhance(factor = 0.1)
        img_sha_l = ImageEnhance.Sharpness(img).enhance(factor = 1.9)

        finished_patches = deal_patches(img_rot180, patch_size, patch_path, svs_name, 'F')
        print(',k=F: ', finished_patches, end=' ')
        finished_patches = deal_patches(img_rot270, patch_size, patch_path, svs_name, 'G')
        print(',k=G: ', finished_patches, end=' ')
        finished_patches = deal_patches(img_rot135, patch_size, patch_path, svs_name, 'H')
        print(',k=H: ', finished_patches, end=' ')
        finished_patches = deal_patches(img_rot225, patch_size, patch_path, svs_name, 'I')
        print(',k=I: ', finished_patches, end=' ')
        finished_patches = deal_patches(img_rot315, patch_size, patch_path, svs_name, 'J')
        print(',k=J: ', finished_patches, end=' ')
        finished_patches = deal_patches(img_bri_h, patch_size, patch_path, svs_name, 'K')
        print(',k=K: ', finished_patches, end=' ')
        finished_patches = deal_patches(img_bri_l, patch_size, patch_path, svs_name, 'L')
        print(',k=L: ', finished_patches)
        finished_patches = deal_patches(img_col_h, patch_size, patch_path, svs_name, 'M')
        print(',k=M: ', finished_patches, end=' ')
        finished_patches = deal_patches(img_col_l, patch_size, patch_path, svs_name, 'N')
        print(',k=N: ', finished_patches)
        finished_patches = deal_patches(img_con_h, patch_size, patch_path, svs_name, 'O')
        print(',k=O: ', finished_patches, end=' ')
        finished_patches = deal_patches(img_con_l, patch_size, patch_path, svs_name, 'P')
        print(',k=P: ', finished_patches)
        finished_patches = deal_patches(img_sha_h, patch_size, patch_path, svs_name, 'Q')
        print(',k=Q: ', finished_patches, end=' ')
        finished_patches = deal_patches(img_sha_l, patch_size, patch_path, svs_name, 'R')
        print(',k=R: ', finished_patches)

        end_time = datetime.datetime.now()
        time = str((end_time - start_time).seconds)
        print('with ' + time + ' seconds, ' + 'finished ' + svs_name + ' in ' +
              str(datetime.datetime.now()).split('.')[0])

    print('Extract Finish!')


if __name__ == '__main__':
    img_path = './WSI/'
    patch_path = './Patch/'
    get_patches(img_path, patch_path, 256)