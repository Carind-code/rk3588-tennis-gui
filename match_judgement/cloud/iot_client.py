from linkkit import linkkit as sdk
import logging
import json

class AliyunIoTClient:
    def __init__(self, conf):
        self.conf = conf
        self.lk = sdk.LinkKit(
            host_name=conf['host_name'],
            product_key=conf['product_key'],
            device_name=conf['device_name'],
            device_secret=conf['device_secret']
        )
        self.connected = False
        self.lk.on_connect = self._on_connect

    def _on_connect(self, session_flag, rc, userdata): # 函数名前加下划线表示是内部使用，主程序不直接调用
        if rc == 0:
            self.connected = True
            logging.info("云端连接成功")
        # 这里删去了原版的处理rc!=0的情况和on_disconnect()，把连接失败的情况交给self.lk.start_worker_loop()处理，它会不断尝试重连

    def connect(self):
        self.lk.connect_async() # 改用阿里云提供的函数，发起请求（异步），即只发送不管有没有连上，当收到同意，阿里云SDK会自动调用_on_connect()
        self.lk.start_worker_loop() # 改用阿里云提供的函数，驱动后台运行，自动处理心跳包、断线重连、消息确认。

    def send_data(self, payload_dict):
        """发送 JSON 数据"""
        payload = json.dumps(payload_dict, ensure_ascii=False)
        return self.lk.publish_topic(self.conf['topic'], payload, qos=1) # 阿里云库里专门负责发消息的函数
