# OV13855固定焦距

板端OV13855配套DW9763为可变焦镜头。为避免连续自动对焦搜焦：

- IQ文件将AF模式改为`CalibDbV2_AFMODE_FIXED`。
- 关闭`contrast_af`和`video_contrast_af`。
- `qiuwu-fixed-focus.service`在RKAIQ启动后，将镜头固定为`focus_absolute=40`。

当前焦点值为`40`，范围为`0-64`。若实际画面不够清晰，可在板端修改服务中的值后执行：

```bash
sudo systemctl daemon-reload
sudo systemctl restart qiuwu-fixed-focus.service
```

恢复连续自动对焦：恢复`/etc/iqfiles/ov13855_CMK-OT2016-FV1_default.autofocus-20260721.json`为默认IQ文件，禁用该服务并重启`rkaiq_3A.service`。
