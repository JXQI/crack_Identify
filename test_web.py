from flask import Flask
from flask import jsonify
from flask import request, render_template
from Mul_models import models_select
import torch
from torchvision import transforms
import settings
from PIL import Image
# Platform.
import platform
print(platform.python_version())
if ('2.' in platform.python_version()):
    from StringIO import StringIO as Bytes2Data
else:
    from io import BytesIO as Bytes2Data

#模型加载
Net = models_select(class_num=2)
net = Net.net('ResNet50')
weight_path='./Weights/best_ResNet50_1_99.pth'
net.load_state_dict(torch.load(weight_path,map_location=torch.device('cpu')))
net.eval()

#增加web后端
app = Flask(__name__)
@app.route('/', methods=['GET'])
def index():
    """
    首页，vue入口
    """
    return render_template('index.html')

@app.route('/api/v1/pets_classify/', methods=['POST'])
def pets_classify():
    img = request.files.get('file').read()
    img=Image.open(Bytes2Data(img))
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img = transform(img)
    img = img.unsqueeze(0)
    output = net(img)
    _, pet_class = torch.max(output, 1)
    pet_cls_prob=0 #TODO:
    res = {
        'code': 0,
        'data': {
            'pet_cls': int(pet_class),
            'probability': pet_cls_prob,
            'msg': '<br><br><strong style="font-size: 48px;">{}</strong> <span style="font-size: 24px;"'
                   '>概率<strong>{}</strong></span>'.format(int(pet_class), pet_cls_prob),
        }
    }
    # 返回json数据
    return jsonify(res)
if __name__ == '__main__':
    app.run(port=settings.WEB_PORT)