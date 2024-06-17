import replicate
import urllib.request
from PIL import Image
import hou
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def test():
    model_url="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"

    pre_prompt = "You are a Houdini AI assistant, please respond everything in SideFX Houdini context"
    #pre_prompt = ""
    #usr_prompt = "How can I crate an OSL shader in Houdini?"
    usr_prompt = hou.ui.readInput("Enter a prompt: ")

    response = replicate.run(model_url,
                             input={"prompt": f"{pre_prompt} {usr_prompt} AI: ", "temperature": 0.1, "max_length": 500, "repetition_penalty": 1}
                             )

    res = ""

    for r in response:
      res += r

    print(res)


def textureGenerator():

    usr_input = hou.ui.readInput("Enter a prompt: ")
    output = replicate.run(
        "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        input={
            "prompt": f"{usr_input} ,high resolution, realistic, tileable"
        }
    )

    res = output[0]
    print(res)
    urllib.request.urlretrieve(res, "/Users/felipepesantez/Documents/development/python/rway_dev/repos/houdiniUtils.git/final_week/replicate_api/texture01.png")

    texture_node = hou.selectedNodes()[0]
    texture_node.parm("map").set("/Users/felipepesantez/Documents/development/python/rway_dev/repos/houdiniUtils.git/final_week/replicate_api/texture01.png")



