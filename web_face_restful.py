#encoding:utf-8
# WEB后端服务程序
# 基于Python Flask 生成 RestfulAPI

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort, Response
import time
import os
import uuid
import base64
import json
import numpy as np
import facenet
import tensorflow as tf
import align.detect_face
from scipy import misc

app = Flask(__name__)
UPLOAD_FOLDER = 'upload'
CROP_IMG_FOLDER = 'crop_faces'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CROP_IMG_FOLDER'] = CROP_IMG_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG'])

# 创建load_and_align_data网络
print('Creating networks and loading parameters')
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

        # 加载模型
        model='./20180402-114759/'
        facenet.load_model(model)

        # 获取输入输出tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")


# 允许上传文件类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# JSONEncoder 重写
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

# 检测人脸位置
def DetectFaceLocation(ImagePath):
    minsize = 20 # 最小人脸尺寸
    threshold = [ 0.6, 0.7, 0.7 ]  # 阈值
    factor = 0.709

    ImgArray = misc.imread(os.path.expanduser(ImagePath), mode='RGB')
    bounding_boxes, _ = align.detect_face.detect_face(ImgArray, minsize, pnet, rnet, onet, threshold, factor)
    det_result = bounding_boxes.astype(int)
    if len(det_result) < 1:
        return False, None
    return True, det_result

# 生成人脸向量模型
def EmbeddingFace(ImagePath):
    images = []
    img = misc.imread(ImagePath, mode='RGB')
    image = facenet.prewhiten(img)
    images.append(image)
    img_stack = np.stack(images)
    feed_dict = { images_placeholder: img_stack, phase_train_placeholder:False }
    img_emb = sess.run(embeddings, feed_dict=feed_dict)
    return img_emb

# 人脸剪切对齐
def ImgFaceCrop(image_path, new_image_size, face_location):
    # 读取图片
    img = misc.imread(image_path, mode='RGB')
    img_size = np.asarray(img.shape)[0:2]
    det = face_location
    det[:,0] = np.maximum(det[:,0], 0)
    det[:,1] = np.maximum(det[:,1], 0)
    det[:,2] = np.minimum(det[:,2], img_size[1])
    det[:,3] = np.minimum(det[:,3], img_size[0])
    
    # face剪切对齐后保存文件夹
    crop_dir = os.path.join(basedir, app.config['CROP_IMG_FOLDER'])
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)
    # 原图的人脸名称及位置
    faces = []
    for i in range(len(face_location)):
        temp_crop = img[det[i,1]:det[i,3], det[i,0]:det[i,2], :]
        aligned = misc.imresize(temp_crop, (new_image_size, new_image_size), interp='bilinear')
        crop_name_pre = str(uuid.uuid1())
        crop_name = crop_name_pre + '.jpg'
        new_img_path = os.path.join(crop_dir, crop_name)
        misc.imsave(new_img_path, aligned)
        faces.append({'face_name': crop_name,'top_left':(face_location[i,0],face_location[i,1]), 'bottom_right': (face_location[i,2], face_location[i,3])})
    return faces

# 加载上传文件页面
@app.route('/upload')
def upload():
    return render_template('up.html')
 
# 上传图片
@app.route('/up_photo', methods=['POST'], strict_slashes=False)
def up_photo():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['photo']
    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        print("Saving file " + fname + "...")
        ext = fname.rsplit('.', 1)[1]
        new_filename = str(uuid.uuid1()) + '.' + ext
        new_file_path = os.path.join(file_dir, new_filename)
        f.save(new_file_path)
        print("Save " + new_filename + " sucess!")
        return jsonify({"sucess": True, "file_name": new_filename})
    else:
        # 错误码1001-文件上传失败
        return jsonify({"sucess": False, "error": 1001})

# 检测人脸位置
@app.route('/detect/<string:filename>', methods=['GET'])
def detect(filename):
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if request.method == 'GET':
        if filename is None:
            # 错误码1002-图片名为空
            return jsonify({"sucess": False, "error": 1002})
        else:
            file_path = os.path.join(file_dir, '%s' % filename)
            if os.path.exists(file_path):
                status, ImgFaceLoction = DetectFaceLocation(file_path)
                if status:
                    FaceNumbers = len(ImgFaceLoction)
                    faces = ImgFaceCrop(file_path, 160, ImgFaceLoction)
                else:
                    FaceNumbers = 0
                    faces = None
                return Response(json.dumps({"sucess": True,"file_name": filename ,"face_numbers": FaceNumbers, "faces": faces}, cls=MyEncoder), mimetype='application/json')
            else:
                # 错误码1003-图片文件不存在
                return jsonify({"sucess": False, "error": 1003, "error_msg": "file not exist!"})
    else:
        pass
    
# 生成人脸128维face_token
@app.route('/embedding/<string:filename>', methods=['GET'])
def embedding(filename):
    file_dir = os.path.join(basedir, app.config['CROP_IMG_FOLDER'])
    if request.method == 'GET':
        if filename is None:
            # 错误码1002-图片名为空
            return jsonify({"sucess": False, "error": 1002})
        else:
            file_path = os.path.join(file_dir, '%s' % filename)
            if os.path.exists(file_path):
                face_token = EmbeddingFace(file_path)[0]
                return Response(json.dumps({"sucess": True, "filename": filename, "face_token": face_token.tolist()}), mimetype='application/json')
            else:
                # 错误码1003-图片文件不存在
                return jsonify({"sucess": False, "error": 1003, "error_msg": "file not exist!"})
    else:
        pass

# 下载图片
@app.route('/download/<string:filename>', methods=['GET'])
def download(filename):
    if request.method == "GET":
        if os.path.isfile(os.path.join('upload', filename)):
            return send_from_directory('upload', filename, as_attachment=True)
        pass

# 显示图片
@app.route('/show_photo/<string:filename>', methods=['GET'])
def show_photo(filename):
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass
 
if __name__ == '__main__':
    # 开启Resultful服务
    app.run(debug=True)
    # r = EmbeddingFace("./crop_faces/a084c9ac-1afe-11e9-befa-94e979b133dc.jpg")
