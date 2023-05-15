import os
import six
import imageio
import random
from PIL import Image
import numpy as np

from datasetslib.images import ImagesDataset
from datasetslib.utils import imutil
from datasetslib import util
from datasetslib import datasets_root


try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip

vgg_image_size = 224
inception_image_size = 299


class CRocks(ImagesDataset):
    def __init__(self):
        ImagesDataset.__init__(self)
        self.dataset_name = 'pluton-rocks'   #'intrusive-rocks'
        self.source_url = 'http://www.ags.org.cn/'   # 临时设置的数据下载url地址
        self.source_files = ['pluton-rocks.zip']
        self.dataset_home = os.path.join(datasets_root, self.dataset_name)
        self.height = None
        self.width = None
        self.depth = None
        self.id2label = {}
        self.label2id = {}

        self.x_layout = imutil.LAYOUT_NCHW
        self.x_layout_file = imutil.LAYOUT_NCHW

        self.n_classes = 7
        # 设置是否居中裁剪图像, True:居中裁剪
        self.IsCenterCrop = True

        # 数据增强后的存储文件夹
        self.augudata_name = 'plutonrocks_augu'
        # 数据通道融合后的存储文件夹
        self.mergedata_name = 'plutonrocks_merge'

    def load_data(self, force=False, shuffle=True, x_is_images=False):


        # 根据岩石训练图像总数修改 10类各100张
        n_train = util.countPathFileNum(os.path.join(self.dataset_home, 'train'))
        # 根据岩石验证图像总数修改 10类各20张
        n_valid = util.countPathFileNum(os.path.join(self.dataset_home, 'val'))

        x_train_files = []
        y_train = np.zeros((n_train,), dtype=np.uint8)
        x_valid_files = []
        y_valid=np.zeros((n_valid,), dtype=np.uint8)

        n_trainnum = 0
        n_valnum = 0

        for i in range(self.n_classes):
            label = id2label[i]
            ifolder = os.path.join(self.dataset_home, 'train', label)
            print(os.listdir(ifolder))
            files = [name for name in os.listdir(ifolder) if name.endswith('.jpg')]
            for f in files:
                x_train_files.append(os.path.join(ifolder, f))
                y_train[n_trainnum] = i
                n_trainnum += 1
            #y_train[i * (n_train // self.n_classes): (i+1) * (n_train // self.n_classes)] = i

            ifolder = os.path.join(self.dataset_home, 'val', label)
            files = [name for name in os.listdir(ifolder) if name.endswith('.jpg')]
            for f in files:
                x_valid_files.append(os.path.join(ifolder, f))
                y_valid[n_valnum] = i
                n_valnum += 1
            #y_valid[i * (n_valid // self.n_classes): (i+1) * (n_valid // self.n_classes)] = i

        if shuffle:
            x_train_files, y_train = self.shuffle_xy(x_train_files, y_train)

        if x_is_images:
            x_train = self.load_images(x_train_files)
            x_valid = self.load_images(x_valid_files)
        else:
            x_train = x_train_files
            x_valid = x_valid_files
        self.x_is_images = x_is_images

        self.part['X_train'] = x_train
        y_train = y_train.astype('int')
        self.part['Y_train'] = y_train

        self.part['X_valid'] = x_valid
        y_valid = y_valid.astype('int')
        self.part['Y_valid'] = y_valid

        return x_train, y_train, x_valid, y_valid

    def load_mergedata(self, force=False, shuffle=True, x_is_images=False):
        self.dataset_home = os.path.join(datasets_root, self.mergedata_name)
        self.dataset_home = os.path.join(self.dataset_home, self.mergedata_name)
        ilabels = os.listdir(os.path.join(self.dataset_home, 'train'))
        label2id = dict(zip(ilabels, range(len(ilabels))))
        print(label2id)

        self.label2id = label2id
        id2label = dict(zip(label2id.values(), label2id.keys()))
        self.id2label = id2label

        # 根据岩石训练图像处理融合后总数修改 10类各100张
        n_train = util.countPathFileNum(os.path.join(self.dataset_home, 'train'))
        # 根据岩石验证图像处理融合总数修改 10类各20张
        n_valid = util.countPathFileNum(os.path.join(self.dataset_home, 'val'))

        x_train_files = []
        y_train = np.zeros((n_train,), dtype=np.uint8)
        x_valid_files = []
        y_valid=np.zeros((n_valid,), dtype=np.uint8)

        for i in range(self.n_classes):
            label = id2label[i]
            ifolder = os.path.join(self.dataset_home, 'train', label)
            print(os.listdir(ifolder))
            files = [name for name in os.listdir(ifolder) if name.endswith('.jpg')]
            for f in files:
                x_train_files.append(os.path.join(ifolder, f))
            y_train[i * (n_train // self.n_classes): (i+1) * (n_train // self.n_classes)] = i

            ifolder = os.path.join(self.dataset_home, 'val', label)
            files = [name for name in os.listdir(ifolder) if name.endswith('.jpg')]
            for f in files:
                x_valid_files.append(os.path.join(ifolder,f))
            y_valid[i * (n_valid // self.n_classes): (i+1) * (n_valid // self.n_classes)] = i

        if shuffle:
            x_train_files, y_train = self.shuffle_xy(x_train_files, y_train)

        if x_is_images:
            x_train = self.load_images(x_train_files)
            x_valid = self.load_images(x_valid_files)
        else:
            x_train = x_train_files
            x_valid = x_valid_files
        self.x_is_images = x_is_images

        self.part['X_train'] = x_train
        self.part['Y_train'] = y_train

        self.part['X_valid'] = x_valid
        self.part['Y_valid'] = y_valid

        return x_train, y_train, x_valid, y_valid

    def load_auguorresizedata(self, force=False, datafolder='', tagList=['train', 'val'], shuffle=True, x_is_images=False):
        self.dataset_home = os.path.join(datasets_root, datafolder)

        ilabels = os.listdir(os.path.join(self.dataset_home, 'train'))
        label2id = dict(zip(ilabels, range(len(ilabels))))
        print(label2id)
        self.label2id = label2id
        id2label = dict(zip(label2id.values(), label2id.keys()))
        self.id2label = id2label

        # 根据岩石训练图像增强后总数修改
        n_train = util.countPathFileNum(os.path.join(self.dataset_home, 'train'))
        # 根据岩石验证图像总数修改
        n_valid = util.countPathFileNum(os.path.join(self.dataset_home, 'val'))

        x_train_files = []
        y_train = np.zeros((n_train,), dtype=np.uint8)
        x_valid_files = []
        y_valid = np.zeros((n_valid,), dtype=np.uint8)

        # 根据岩石测试图像总数修改
        x_test_files = []
        n_test = 0
        tag = 'test'
        if tag in tagList:
            n_test = util.countPathFileNum(os.path.join(self.dataset_home, tag))
        y_test = np.zeros((n_test,), dtype=np.uint8)

        n_trainnum = 0
        n_valnum = 0
        n_testnum = 0
        for i in range(self.n_classes):
            label = id2label[i]
            ifolder = os.path.join(self.dataset_home, 'train', label)
            # print(os.listdir(ifolder))
            files = [name for name in os.listdir(ifolder) if name.endswith('.jpg')]
            for f in files:
                x_train_files.append(os.path.join(ifolder, f))
                y_train[n_trainnum] = i
                n_trainnum += 1

            ifolder = os.path.join(self.dataset_home, 'val', label)
            files = [name for name in os.listdir(ifolder) if name.endswith('.jpg')]
            for f in files:
                x_valid_files.append(os.path.join(ifolder, f))
                y_valid[n_valnum] = i
                n_valnum += 1

            if tag in tagList:
                ifolder = os.path.join(self.dataset_home, tag, label)
                files = [name for name in os.listdir(ifolder) if name.endswith('.jpg')]
                for f in files:
                    x_test_files.append(os.path.join(ifolder, f))
                    y_test[n_testnum] = i
                    n_testnum += 1

        if shuffle:
            x_train_files, y_train = self.shuffle_xy(x_train_files, y_train)

        if x_is_images:
            x_train = self.load_images(x_train_files)
            x_valid = self.load_images(x_valid_files)
            if n_testnum >= 1:
                x_test = self.load_images(x_test_files)
        else:
            x_train = x_train_files
            x_valid = x_valid_files
            if n_testnum >= 1:
                x_test = x_test_files
        self.x_is_images = x_is_images

        self.part['X_train'] = x_train
        self.part['Y_train'] = y_train

        self.part['X_valid'] = x_valid
        self.part['Y_valid'] = y_valid

        if n_testnum >= 1:
            self.part['X_test'] = x_test
            self.part['Y_test'] = y_test
            return x_train, y_train, x_valid, y_valid, x_test, y_test

        return x_train, y_train, x_valid, y_valid

    def preprocess_for_vgg(self, incoming):
        img_size = vgg_image_size

        height = img_size
        width = img_size

        if isinstance(incoming, six.string_types):
            img = self.load_image(incoming)
        elif isinstance(incoming, imageio.core.util.Image ):
            img = Image.fromarray(incoming)
        else: #
            img = incoming

        #print(type(img))

        img = self.resize_image(img, height, width, self.IsCenterCrop)
        img = self.pil_to_nparray(img)
        if len(img.shape) == 2:   # greyscale or no channels then add three channels
            h = img.shape[0]
            w = img.shape[1]
            img = np.dstack([img]*3)

        means = np.array([[[123.68, 116.78, 103.94]]]) #shape=[1, 1, 3]
        try:
            img = img - means
        except Exception as ex:
            print('Error preprocessing ',incoming)
            print(ex)

        return img

    def preprocess_for_inception(self, incoming):

        img_size = inception_image_size

        height = img_size
        width = img_size

        if isinstance(incoming, six.string_types):
            img = self.load_image(incoming)
        elif isinstance(incoming, imageio.core.util.Image):
            img = Image.fromarray(incoming)
        else:  #
            img = incoming

        # print(type(img))

        img = self.resize_image(img, height, width, self.IsCenterCrop)
        img = self.pil_to_nparray(img)
        if len(img.shape) == 2:  # greyscale or no channels then add three channels
            h = img.shape[0]
            w = img.shape[1]
            img = np.dstack([img] * 3)

        img = ((img / 255.0) - 0.5) * 2.0

        return img

    def preprocess_for_model_custom(self, incoming, cusheight=448, cuswidth=448):
        height = cusheight
        width = cuswidth

        if isinstance(incoming, six.string_types):
            img = self.load_image(incoming)
        elif isinstance(incoming, imageio.core.util.Image ):
            img = Image.fromarray(incoming)
        else: #
            img = incoming

        img = self.resize_image(img, height, width, self.IsCenterCrop)
        img = self.pil_to_nparray(img)
        img = (img / 255.0)
        if len(img.shape) == 2:   # greyscale or no channels then add three channels
            h = img.shape[0]
            w = img.shape[1]
            img = np.dstack([img]*3)

        #means = np.array([[[123.68, 116.78, 103.94]]]) #shape=[1, 1, 3]
        #try:
        #    img = img - means
        #except Exception as ex:
        #    print('Error preprocessing ',incoming)
        #    print(ex)

        return img

    def get_savepath(self, augufolder='', pre='', step=1, angle=0):
        if angle > 0:
            str_path = '%s\\%s_%s_%s.jpg' % (augufolder, step, pre, angle)
        else:
            str_path = '%s\\%s_%s.jpg' % (augufolder, step, pre)
        return str_path


    # 上下和左右镜像岩石图像
    def augumentation_flip(self, image_path='', step=0, filefolder=''):
        image = Image.open(image_path)
        out_horizontal = image.transpose(Image.FLIP_LEFT_RIGHT)
        strtemp = self.get_savepath(augufolder=filefolder, pre='hflip', step=step)
        out_horizontal = self.resize_image(out_horizontal, self.height, self.height, False)
        out_horizontal.save(strtemp)

        out_vetical = image.transpose(Image.FLIP_TOP_BOTTOM)
        strtemp = self.get_savepath(augufolder=filefolder, pre='vflip', step=step)
        out_vetical = self.resize_image(out_vetical, self.height, self.height, False)
        out_vetical.save(strtemp)

    # 旋转增强岩石图像8张，0为复制原图像，其它在1-4、5-8、9-12、13-16、17-20、21-24、25-30间随机生成旋转角度
    def augumentation_rotate(self, image_path='', step=0, filefolder=''):
        image = Image.open(image_path)

        random_angle = 0
        out0 = image.rotate(random_angle)
        strtemp = self.get_savepath(augufolder=filefolder, pre='rotate', step=step, angle=random_angle)
        out0 = self.resize_image(out0, self.height, self.height, False)
        out0.save(strtemp)

        # 随机旋转图像 0-30范围内
        random_angle = np.random.randint(1, 4)
        out1 = image.rotate(random_angle)
        strtemp = self.get_savepath(augufolder=filefolder, pre='rotate', step=step, angle=random_angle)
        out1 = self.resize_image(out1, self.height, self.height, False)
        out1.save(strtemp)

        random_angle = np.random.randint(5, 8)
        out2 = image.rotate(random_angle)
        strtemp = self.get_savepath(augufolder=filefolder, pre='rotate', step=step, angle=random_angle)
        out2 = self.resize_image(out2, self.height, self.height, False)
        out2.save(strtemp)

        random_angle = np.random.randint(9, 12)
        out3 = image.rotate(random_angle)
        strtemp = self.get_savepath(augufolder=filefolder, pre='rotate', step=step, angle=random_angle)
        out3 = self.resize_image(out3, self.height, self.height, False)
        out3.save(strtemp)

        random_angle = np.random.randint(13, 16)
        out4 = image.rotate(random_angle)
        strtemp = self.get_savepath(augufolder=filefolder, pre='rotate', step=step, angle=random_angle)
        out4 = self.resize_image(out4, self.height, self.height, False)
        out4.save(strtemp)

        random_angle = np.random.randint(17, 20)
        out5 = image.rotate(random_angle)
        strtemp = self.get_savepath(augufolder=filefolder, pre='rotate', step=step, angle=random_angle)
        out5 = self.resize_image(out5, self.height, self.height, False)
        out5.save(strtemp)

        random_angle = np.random.randint(21, 24)
        out6 = image.rotate(random_angle)
        strtemp = self.get_savepath(augufolder=filefolder, pre='rotate', step=step, angle=random_angle)
        out6 = self.resize_image(out6, self.height, self.height, False)
        out6.save(strtemp)

        random_angle = np.random.randint(25, 30)
        out7 = image.rotate(random_angle)
        strtemp = self.get_savepath(augufolder=filefolder, pre='rotate', step=step, angle=random_angle)
        out7 = self.resize_image(out7, self.height, self.height, False)
        out7.save(strtemp)

    # 上下和左右镜像岩石图像
    def augumentation_crop_flip(self, image_path='', step=0, filefolder=''):
        (path, filename) = os.path.split(image_path)
        (file, ext) = os.path.splitext(filename)
        hprestr = '%s_%s' % (file, 'hflip')
        vprestr = '%s_%s' % (file, 'vflip')

        image = Image.open(image_path)
        out_horizontal = image.transpose(Image.FLIP_LEFT_RIGHT)
        strtemp = self.get_savepath(augufolder=filefolder, pre=hprestr, step=step)
        out_horizontal = self.resize_image(out_horizontal, self.height, self.height, False)
        out_horizontal.save(strtemp)

        out_vetical = image.transpose(Image.FLIP_TOP_BOTTOM)
        strtemp = self.get_savepath(augufolder=filefolder, pre=vprestr, step=step)
        out_vetical = self.resize_image(out_vetical, self.height, self.height, False)
        out_vetical.save(strtemp)

    def augumentation_crop_rotate(self, image_path='', step=0, filefolder=''):
        (path, filename) = os.path.split(image_path)
        (file, ext) = os.path.splitext(filename)
        prestr = '%s_%s' % (file, 'crop')

        image = Image.open(image_path)

        # 随机旋转图像 0-30范围内
        random_angle = np.random.randint(1, 4)
        out1 = image.rotate(random_angle)
        strtemp = self.get_savepath(augufolder=filefolder, pre=prestr, step=step, angle=random_angle)
        out1 = self.resize_image(out1, self.height, self.height, False)
        out1.save(strtemp)

        random_angle = np.random.randint(5, 8)
        out2 = image.rotate(random_angle)
        strtemp = self.get_savepath(augufolder=filefolder, pre=prestr, step=step, angle=random_angle)
        out2 = self.resize_image(out2, self.height, self.height, False)
        out2.save(strtemp)

        random_angle = np.random.randint(9, 12)
        out3 = image.rotate(random_angle)
        strtemp = self.get_savepath(augufolder=filefolder, pre=prestr, step=step, angle=random_angle)
        out3 = self.resize_image(out3, self.height, self.height, False)
        out3.save(strtemp)

        random_angle = np.random.randint(13, 16)
        out4 = image.rotate(random_angle)
        strtemp = self.get_savepath(augufolder=filefolder, pre=prestr, step=step, angle=random_angle)
        out4 = self.resize_image(out4, self.height, self.height, False)
        out4.save(strtemp)

        random_angle = np.random.randint(17, 20)
        out5 = image.rotate(random_angle)
        strtemp = self.get_savepath(augufolder=filefolder, pre=prestr, step=step, angle=random_angle)
        out5 = self.resize_image(out5, self.height, self.height, False)
        out5.save(strtemp)

        random_angle = np.random.randint(21, 24)
        out6 = image.rotate(random_angle)
        strtemp = self.get_savepath(augufolder=filefolder, pre=prestr, step=step, angle=random_angle)
        out6 = self.resize_image(out6, self.height, self.height, False)
        out6.save(strtemp)

        random_angle = np.random.randint(25, 30)
        out7 = image.rotate(random_angle)
        strtemp = self.get_savepath(augufolder=filefolder, pre=prestr, step=step, angle=random_angle)
        out7 = self.resize_image(out7, self.height, self.height, False)
        out7.save(strtemp)

    # 随机裁剪岩石图像
    def augumentation_crop(self, image_path='', step=0, filefolder=''):
        image = Image.open(image_path)
        if image.size[0] > image.size[1]:
            crop_width = image.size[1] // 3 * 2
        else:
            crop_width = image.size[0] // 3 * 2

        image_dst_crop_1 = self.random_crop(image, [crop_width, crop_width], padding=10)
        strtemp_1 = self.get_savepath(augufolder=filefolder, pre='crop', step=step, angle=1)
        image_dst_crop_1 = self.resize_image(image_dst_crop_1, self.height, self.height, False)
        image_dst_crop_1.save(strtemp_1)
        self.augumentation_crop_rotate(image_path=strtemp_1, step=step, filefolder=filefolder)
        self.augumentation_crop_flip(image_path=strtemp_1, step=step, filefolder=filefolder)

        image_dst_crop_2 = self.random_crop(image, [crop_width, crop_width], padding=10)
        strtemp_2 = self.get_savepath(augufolder=filefolder, pre='crop', step=step, angle=2)
        image_dst_crop_2 = self.resize_image(image_dst_crop_2, self.height, self.height, False)
        image_dst_crop_2.save(strtemp_2)
        self.augumentation_crop_rotate(image_path=strtemp_2, step=step, filefolder=filefolder)
        self.augumentation_crop_flip(image_path=strtemp_2, step=step, filefolder=filefolder)

        image_dst_crop_3 = self.random_crop(image, [crop_width, crop_width], padding=10)
        strtemp_3 = self.get_savepath(augufolder=filefolder, pre='crop', step=step, angle=3)
        image_dst_crop_3 = self.resize_image(image_dst_crop_3, self.height, self.height, False)
        image_dst_crop_3.save(strtemp_3)
        self.augumentation_crop_rotate(image_path=strtemp_3, step=step, filefolder=filefolder)
        self.augumentation_crop_flip(image_path=strtemp_3, step=step, filefolder=filefolder)

        image_dst_crop_4 = self.random_crop(image, [crop_width, crop_width], padding=10)
        strtemp_4 = self.get_savepath(augufolder=filefolder, pre='crop', step=step, angle=4)
        image_dst_crop_4 = self.resize_image(image_dst_crop_4, self.height, self.height, False)
        image_dst_crop_4.save(strtemp_4)
        self.augumentation_crop_rotate(image_path=strtemp_4, step=step, filefolder=filefolder)
        self.augumentation_crop_flip(image_path=strtemp_4, step=step, filefolder=filefolder)

    def data_augumentation(self, auguPath='', taglist=['train', 'val']):
        data_augu_home = os.path.join(datasets_root, auguPath)
        id2label = self.id2label
        for tag in taglist:
            for i in range(self.n_classes):
                label = id2label[i]
                ifolder = os.path.join(self.dataset_home, tag, label)
                augufolder = os.path.join(data_augu_home, tag, label)
                if not os.path.exists(augufolder):
                    os.makedirs(augufolder)
                print(os.listdir(ifolder))
                files = [name for name in os.listdir(ifolder) if name.endswith('.jpg')]
                filenum = 0
                for f in files:
                    f_path = os.path.join(ifolder, f)
                    self.augumentation_rotate(image_path=f_path, step=filenum, filefolder=augufolder)
                    self.augumentation_flip(image_path=f_path, step=filenum, filefolder=augufolder)
                    self.augumentation_crop(image_path=f_path, step=filenum, filefolder=augufolder)
                    filenum +=1

    def random_crop(self, image, crop_shape, padding=None):
        oshape = image.size
        if padding:
            oshape_pad = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
            img_pad = Image.new("RGB", (oshape_pad[0], oshape_pad[1]))
            img_pad.paste(image, (padding, padding))

            nh = np.random.randint(0, oshape_pad[0] - crop_shape[0])
            nw = np.random.randint(0, oshape_pad[1] - crop_shape[1])
            image_crop = img_pad.crop((nh, nw, nh + crop_shape[0], nw + crop_shape[1]))

            return image_crop
        else:
            print("WARNING!!! nothing to do!!!")
            return image


    def data_resize_custom(self, resize_path='', auguPath='', cusheight=448, cuswidth=448):
        data_resize_home = os.path.join(datasets_root, resize_path)
        #data_resize_home = os.path.join(data_resize_home, resize_path)

        data_augu_home = os.path.join(datasets_root, auguPath)

        id2label = self.id2label
        for i in range(self.n_classes):
            label = id2label[i]
            ifolder = os.path.join(data_augu_home, 'train', label)
            resizefolder = os.path.join(data_resize_home, 'train', label)
            if not os.path.exists(resizefolder):
                os.makedirs(resizefolder)
            # print(os.listdir(ifolder))
            print('Starting resize {}:{} from folder {} to {} '.format(i, label, ifolder, resizefolder))
            files = [name for name in os.listdir(ifolder) if name.endswith('.jpg')]
            for f in files:
                f_path = os.path.join(ifolder, f)
                s_path = os.path.join(resizefolder, f)
                image = Image.open(f_path)
                if image.size[0] > image.size[1]:
                    height = image.size[1]
                else:
                    height = image.size[0]
                img = self.resize_image(image, height, height, crop_or_pad=True)
                simg = self.resize_image(img, cusheight, cuswidth, False)
                simg.save(s_path)

            ifolder = os.path.join(data_augu_home, 'val', label)
            resizefolder = os.path.join(data_resize_home, 'val', label)
            if not os.path.exists(resizefolder):
                os.makedirs(resizefolder)
            # print(os.listdir(ifolder))
            print('Starting resize {}:{} from folder {} to {} '.format(i, label, ifolder, resizefolder))
            files = [name for name in os.listdir(ifolder) if name.endswith('.jpg')]
            for f in files:
                f_path = os.path.join(ifolder, f)
                s_path = os.path.join(resizefolder, f)
                image = Image.open(f_path)
                if image.size[0] > image.size[1]:
                    height = image.size[1]
                else:
                    height = image.size[0]
                img = self.resize_image(image, height, height, crop_or_pad=True)
                simg = self.resize_image(img, cusheight, cuswidth, False)
                simg.save(s_path)

    def data_random_select(self, random_path='', num_train=500, num_val=50):
        data_random_home = os.path.join(datasets_root, random_path)
        data_home = self.dataset_home
        id2label = self.id2label

        for i in range(self.n_classes):
            label = id2label[i]
            ifolder = os.path.join(data_home, 'train', label)
            randomfolder = os.path.join(data_random_home, 'train', label)
            if not os.path.exists(randomfolder):
                os.makedirs(randomfolder)
            print(os.listdir(ifolder))
            files = [name for name in os.listdir(ifolder) if name.endswith('.jpg')]
            files_random = random.sample(files, num_train)
            for f in files_random:
                f_path = os.path.join(ifolder, f)
                s_path = os.path.join(randomfolder, f)
                image = Image.open(f_path)
                if image.size[0] > image.size[1]:
                    height = image.size[1]
                else:
                    height = image.size[0]
                simg = self.resize_image(image, height, height, crop_or_pad=True)
                simg.save(s_path)

            ifolder = os.path.join(data_home, 'val', label)
            randomfolder = os.path.join(data_random_home, 'val', label)
            if not os.path.exists(randomfolder):
                os.makedirs(randomfolder)
            print(os.listdir(ifolder))
            files = [name for name in os.listdir(ifolder) if name.endswith('.jpg')]
            files_random = random.sample(files, num_val)
            for f in files_random:
                f_path = os.path.join(ifolder, f)
                s_path = os.path.join(randomfolder, f)
                image = Image.open(f_path)
                if image.size[0] > image.size[1]:
                    height = image.size[1]
                else:
                    height = image.size[0]
                simg = self.resize_image(image, height, height, crop_or_pad=True)
                simg.save(s_path)