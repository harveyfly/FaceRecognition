# FaceRecognition
Facenet-based face recognition, web API was created by Python Flask

## Facenet
+ [facenet](https://github.com/davidsandberg/facenet)是谷歌的一篇很有名的论文和开源项目，其实现了将输入的人像最终转换为shape为1*128的向量，然后通过计算不同照片之间的欧几里得距离来判断他们的相似度，当然其中还包含许多技巧以及创新的想法，最终的在lfw人脸信息数据库上的准确率达到99%+++

## Flask
+ [Flask](http://flask.pocoo.org/docs/1.0/)是一个使用Python编写的轻量级Web应用框架，基于Werkzeug WSGI工具箱和Jinja2 模板引擎。可以使用python flask框架搭建后端服务程序，提供Restful API

## Web Face API

![架构图](https://yespace.xyz/images/tensorflow_server.jpg)

### Detect API

#### 描述
传入图片进行人脸检测，可以检测图片内的所有人脸（目前设置最小值为30×30 pix），对于每个检测出的人脸，会给出其唯一标识 face_token，可用于后续的人脸分析、人脸比对等操作。每个 face_token 以及对应的人脸信息通过 model 计算出来的1*128的向量，作为一个数据元保存在临时存储区（服务重启后该face_token失效），需要保存的face_token可以通过addface API存入数据库中。

#### 图片要求
图片格式：JPG(JPEG)，PNG
图片像素尺寸：最小 48*48 像素，最大 4096*4096 像素
图片文件大小：2 MB

#### 调用URL
[http://localhost:5000/detect](http://localhost:5000/detect)

#### 调用方法
`POST`

#### 请求参数
是否必选|参数名|类型|参数说明
:---|:---|:---|:---
必选|image_file|File|图片，二进制文件，需要用post multipart/form-data的方式上传

#### 返回值说明
字段|类型|说明
:---|:---|:---
sucess|String|标志位，表示请求是否成功
face_num|Int|图片中人脸的数量
faces|Array|被检测出的人脸数组，具体包含内容见下文。注：如果没有检测出人脸则为空数组

#### faces 数组中单个元素的结构
字段|类型|说明
:---|:---|:---
face_token|String|人脸信息在整个系统中的唯一标识，24位随机生成字符串
face_name|String|生成人脸图片的名称，保存在CROP_FACES文件夹中
top_left|Object|人脸位置矩形边框的左上角坐标，数据类型为元组，如：(1.0, 1.0)
bottom_right|Object|人脸位置矩形边框的右下角坐标

### Search API

#### 描述
将两个人脸进行比对，根据距离阈值判断是否为同一个人


