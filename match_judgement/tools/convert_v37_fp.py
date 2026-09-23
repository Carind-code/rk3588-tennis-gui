from rknn.api import RKNN


ONNX_MODEL = "model_v37_360x640_b1_sigmoid.onnx"
RKNN_MODEL = "model_v37_360x640_b1_sigmoid_fp.rknn"


def check(ret, step):
    if ret != 0:
        raise RuntimeError("{} failed: {}".format(step, ret))


def main():
    rknn = RKNN(verbose=True)
    check(
        rknn.config(
            mean_values=[[0] * 9],
            std_values=[[255] * 9],
            target_platform="rk3588",
        ),
        "config",
    )
    check(rknn.load_onnx(model=ONNX_MODEL), "load_onnx")
    check(rknn.build(do_quantization=False), "build")
    check(rknn.export_rknn(RKNN_MODEL), "export_rknn")
    rknn.release()
    print("RKNN exported: {}".format(RKNN_MODEL))


if __name__ == "__main__":
    main()
