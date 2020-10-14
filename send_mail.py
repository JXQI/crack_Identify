# #!/usr/bin/python
# # coding:utf-8

#!/usr/bin/python3
'''
import smtplib
from email.mime.text import MIMEText
from email.header import Header
# 第三方 SMTP 服务
mail_host="smtp.163.com"  #设置服务器
mail_user="18811610296@163.com"    #用户名
mail_pass="VWTEZUGJIHVODSAS"   #口令
sender = '18811610296@163.com'
receivers = ['2390139915@qq.com']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱
message = MIMEText('Python 邮件发送测试...', 'plain', 'utf-8')
message['From'] = Header("W3Cschool教程", 'utf-8')
message['To'] =  Header("测试", 'utf-8')
subject = 'Python SMTP 邮件测试'
message['Subject'] = Header(subject, 'utf-8')
try:
    smtpObj = smtplib.SMTP()
    smtpObj.connect(mail_host, 465)    # 25 为 SMTP 端口号
    smtpObj.login(mail_user,mail_pass)
    smtpObj.sendmail(sender, receivers, message.as_string())
    print ("邮件发送成功")
except smtplib.SMTPException:
    print ("Error: 无法发送邮件")
'''
import smtplib
from email.mime.text import MIMEText
# 引入smtplib和MIMEText
from time import sleep

def sentemail():
    host = 'smtp.163.com'
    # 设置发件服务器地址
    port = 465
    # 设置发件服务器端口号。注意，这里有SSL和非SSL两种形式，现在一般是SSL方式
    sender = '18811610296@163.com'
    # 设置发件邮箱，一定要自己注册的邮箱
    pwd = 'VWTEZUGJIHVODSAS'
    # 设置发件邮箱的授权码密码，根据163邮箱提示，登录第三方邮件客户端需要授权码
    mailto = ['2390139915@qq.com','18811610296@163.com','2403454992@qq.com','2361753941@qq.com']
    # 设置邮件接收人，可以是QQ邮箱
    body = '<h1>模型训练完成</h1><p>你是个大傻子,能不能猜到我是谁?</p>'
    # 设置邮件正文，这里是支持HTML的
    msg = MIMEText(body, 'html')
    # 设置正文为符合邮件格式的HTML内容
    msg['subject'] = '打卡通知'
    # 设置邮件标题
    msg['from'] = sender
    # 设置发送人
    for receiver in mailto:
        msg['to'] = receiver
        # 设置接收人
        try:
            s = smtplib.SMTP_SSL(host, port)
            # 注意！如果是使用SSL端口，这里就要改为SMTP_SSL
            s.login(sender, pwd)
            # 登陆邮箱
            s.sendmail(sender, receiver, msg.as_string())
            # 发送邮件！
            print('Done.sent email success')
        except smtplib.SMTPException:
            print('Error.sent email fail')


if __name__ == '__main__':
    sentemail()
