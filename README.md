# FaceRecognition
Facenet-based face recognition, web API was created by Python Flask

## Facenet
+ [facenet](https://github.com/davidsandberg/facenet)是谷歌的一篇很有名的论文和开源项目，其实现了将输入的人像最终转换为shape为1*128的向量，然后通过计算不同照片之间的欧几里得距离来判断他们的相似度，当然其中还包含许多技巧以及创新的想法，最终的在lfw人脸信息数据库上的准确率达到99%+++

## Flask
+ [Flask](http://flask.pocoo.org/docs/1.0/)是一个使用Python编写的轻量级Web应用框架，基于Werkzeug WSGI工具箱和Jinja2 模板引擎。可以使用python flask框架搭建后端服务程序，提供Restful API

## Web Face API

![架构图](https://yesgithub-1254021701.cos.ap-beijing.myqcloud.com/tf-serving-diagram.svg?q-sign-algorithm=sha1&q-ak=AKIDzlRDMTEUZMHHJPS9jhBvLvnNR7o61ds0&q-sign-time=1553432432;1553434232&q-key-time=1553432432;1553434232&q-header-list=&q-url-param-list=&q-signature=23489964eddece813dc4ac93d802a1aff98f7127&x-cos-security-token=08b5ef72c2f905a79091f945b0dc0715f45fd44f10001)

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
必选|image_file|File|图片，二进制文件，需要用post form-data的方式上传

#### 返回值说明
字段|类型|说明
:---|:---|:---
sucess|Bool|标志位，表示请求是否成功
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
查找相似人脸，根据face_token或者上传的图片文件，以及threshold从数据库的face_set中找到最近距离的人脸信息

#### 图片要求
图片格式：JPG(JPEG)，PNG
图片像素尺寸：最小 48*48 像素，最大 4096*4096 像素
图片文件大小：2 MB

#### 调用URL
[http://localhost:5000/search](http://localhost:5000/search)

#### 调用方法
`POST` `GET`

#### 请求参数
<table>
    <thead>
        <td><strong>是否必选</strong></td>
        <td><strong>参数名</strong></td>
        <td><strong>类型</strong></td>
        <td><strong>参数说明</strong></td>
    </thead>
    <tr>
        <td rowspan="2">必选（二选一）</td>
        <td>face_token</td>
        <td>String</td>
        <td>进行搜索的目标人脸的 face_token ，使用 get 的方法调用</td>
    </tr>
    <tr>
        <td>image_file</td>
        <td>File</td>
        <td>目标人脸所在的图片，二进制文件，需要用 post form-data 的方式上传。</td>
    </tr>
    <tr>
        <td>必选</td>
        <td>threshold</td>
        <td>String</td>
        <td>用于比较的距离阈值, 表示两个人脸信息向量在欧式空间的归一化距离, 越接近0表示相似程度越高（参考值0.65）</td>
    </tr>
</table>

#### 返回值说明
字段|类型|说明
:---|:---|:---
sucess|Bool|标志位，表示操作是否成功
face_token|String|进行搜索的目标人脸的 face_token
threshold|String|用于比较的距离阈值, 表示两个人脸信息向量在欧式空间的归一化距离
cmp_result|Array|人脸搜索的结果，返回 face_set 人脸距离在 threshold 以下的所有人脸信息的集合，具体内容见下文。注：如果没有相似人脸信息则为空数组

#### cmp_result 数组中单个元素的结构
字段|类型|说明
:---|:---|:---
face_token|String|搜索到的人脸 face_token
distant|Float32|目标人脸与 face_set 搜索人脸的距离
face_name|String|搜索人脸的名称（需要在上传人脸信息时提供）

### Add Face API

#### 描述
向 FaceSet 中添加人脸标识 face_token。

#### 调用URL
[http://localhost:5000/addface](http://localhost:5000/addface)

#### 调用方法
`GET`

#### 请求参数
是否必选|参数名|类型|参数说明
:---|:---|:---|:---
必选|face_token|String|需要保存到 face_set 中的 face_token
可选|face_name|String|face_token 对应的人脸名称标识，若参数为空，默认为unknown

#### 返回值说明
字段|类型|说明
:---|:---|:---
sucess|Bool|标志位，表示操作是否成功
face_token|String|保存到face_set中的face_token
face_neme|String|保存到face_set中的face_name

### Remove Face API

#### 描述
从 FaceSet 中删除人脸标识 face_token。

#### 调用URL
[http://localhost:5000/removeface](http://localhost:5000/removeface)

#### 调用方法
`GET`

#### 请求参数
是否必选|参数名|类型|参数说明
:---|:---|:---|:---
必选|face_token|String|需要从 face_set 中移除的 face_token

#### 返回值说明
字段|类型|说明
:---|:---|:---
sucess|Bool|标志位，表示操作是否成功
face_token|String|从 face_set 中移除的 face_token

### ERROR

#### 常见的错误信息说明

错误码|错误信息|说明
:---|:---|:---
1001|BAD_ARGUMENTS|某个参数解析出错（比如必须是数字，但是输入的是非数字字符串; 或者长度过长，etc.）
1002|MISSING_ARGUMENTS|缺少某个必选参数。
1003|INTERNAL_ERROR|服务器内部错误
1004|FIlE_ERROR|文件错误，上传文件类型不支持
1005|UPLOAD_ERROR|上传错误，参数和数据获取失败
1006|FACE_TOKEN_ERROR|查找不到face_token信息，不能识别face_token
1007|FACE_INFO_ERROR|未检测到人脸信息