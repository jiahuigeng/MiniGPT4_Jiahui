import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_minigptv2

trans_prompt = "translate the following sentence to English:"
senti_prompt = "translate the sentence to English:"
facts_prompt = "tell me facts about the following person:"

import pickle
from PIL import Image
from torchvision import transforms
torch.manual_seed(50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

empty_id = 29871
image_shape = (1, 3, 448, 448)
image_token_len = 256

tp = transforms.ToPILImage()


# def get_embs(model, prompt, img_list):
#     device = img_list[0].device

    # prompt_segs = prompt.split('<ImageHere>')
    # assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    # seg_tokens = [
    #     model.llama_tokenizer(
    #         seg, return_tensors="pt", add_special_tokens=i==0).to(device).input_ids # only add bos to the first seg
    #     for i, seg in enumerate(prompt_segs)
    # ]
    # seg_embs = [model.embed_tokens(seg_t) for seg_t in seg_tokens]
    #
    # mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    # mixed_embs = torch.cat(mixed_embs, dim=1)
    # return mixed_embs
def main(args):
    print('Initializing model')
    cfg = Config(args)

    model_config = cfg.model_cfg
    print(model_config)
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Model Initialization Finished')
    chat_state = CONV_VISION_minigptv2.copy()

    image_tensor = torch.randn(image_shape).to(device).requires_grad_(True)
    prompt = None
    if args.task == "trans":
        prompt = trans_prompt
    elif args.task == "facts":
        prompt = facts_prompt
    elif args.task == "senti":
        prompt = senti_prompt

    input_ids = chat.model.llama_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False).to(device).input_ids

    input_embeds = chat.model.embed_tokens(input_ids).to("cuda")
    empty_embed = chat.model.embed_tokens(torch.tensor([[empty_id]]).to(model.device))
    empty_embeds = empty_embed.repeat(1, image_token_len - input_ids.shape[1], 1)
    padded_input_embeds = torch.cat((empty_embeds, input_embeds), dim=1).to(model.device)

    # upload image




    # img_list = []

    # args.img_path = "view.jpg"
    # for i in range(1):
    #     llm_message = chat.upload_img(args.img_path, chat_state, img_list)
    #     # print(llm_message)
    #
    # # ask a question
    # user_message = "Describe the image."
    # # user_message = "Provide a concise description of the given image."
    # chat.ask(user_message, chat_state)
    # chat.encode_img(img_list)




    if args.task in ["senti", "trans", "facts"]:
        best_loss = 100
        best_idx = 0
        best_tensor = None
        optimizer = optim.Adam([image_tensor], lr=0.1)
        cos_loss_fun = nn.CosineEmbeddingLoss()
        # image_embeds = torch.randn(model.encode_images(image_tensor).shape).to(device).requires_grad_(True)
        # optimizer = optim.Adam([image_embeds], lr=0.1)
        # model.train()
        # for param in model.parameters():
        #     param.requires_grad = False

        for step in range(args.num_steps):  # May need more iterations for Adam
            optimizer.zero_grad()
            image_embeds = chat.model.encode_img(image_tensor)[0]
            loss = None
            if args.mode == "full":
                if args.loss == "l2":
                    diff = image_embeds - padded_input_embeds
                    loss = (diff ** 2).mean()

                elif args.loss == "cosine":
                    target_ones = torch.ones(padded_input_embeds.shape[1]).to("cuda")
                    loss = cos_loss_fun(image_embeds[0], padded_input_embeds[0], target_ones)

                elif args.loss == "both":
                    l2_loss = ((image_embeds - padded_input_embeds) ** 2).mean()
                    target_ones = torch.ones(padded_input_embeds.shape[1]).to("cuda")
                    cos_loss = cos_loss_fun(image_embeds[0], padded_input_embeds[0], target_ones)
                    loss = l2_loss + cos_loss

                loss.backward(retain_graph=True)
                optimizer.step()

            elif args.mode == "part":
                len_prompt_token = input_embeds.shape[1]
                target_ones = torch.ones(padded_input_embeds.shape[1])[-len_prompt_token:].to("cuda")
                part_prompt_embeds = padded_input_embeds[0][-len_prompt_token:].to("cuda")
                part_image_embeds = image_embeds[0][-len_prompt_token:].to("cuda")

                if args.loss == "l2":
                    loss = ((part_image_embeds - part_prompt_embeds) ** 2).mean()
                elif args.loss == "cosine":
                    loss = cos_loss_fun(part_image_embeds, part_prompt_embeds, target_ones)
                elif args.loss == "both":
                    l2_loss = ((part_image_embeds - part_prompt_embeds) ** 2).mean()
                    cos_loss = cos_loss_fun(part_image_embeds, part_prompt_embeds, target_ones)
                    loss = l2_loss + cos_loss
                loss.backward(retain_graph=True)
                optimizer.step()


            if step % 1 == 0:  # Print loss every 10 steps
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_idx = step
                    best_tensor = image_tensor.detach().cpu()

                # if args.pre_set == step:
                #     file_name = f"{args.task}_{args.mode}_{args.loss}_{args.pre_set}_prompt.bin"
                #     pickle.dump(image_tensor.detach().cpu(), open(file_name, "wb"))
                print(f'Step {step}, Loss: {loss.item()}')

            if loss < 1e-4:  # A threshold for convergence
                break

        best_name = f"{args.task}_{args.mode}_{args.loss}_best_{best_idx}_prompt.bin"
        pickle.dump(best_tensor, open(os.path.join("generated_images", best_name), "wb"))
        print(best_loss, best_idx)

    else:
        image_tensor = torch.tensor(pickle.load(open("prompt.bin", "rb"))).to(device="cuda")
        image_embeds = chat.model.encode_img(image_tensor)
        padded_embeds = torch.tensor(pickle.load(open("padded_prompt.bin", "rb"))).to(device="cuda")
        loss = ((image_embeds - padded_embeds)**2).mean()
        print(f'Loss: {loss.item()}')






    # # get answer
    # llm_message = chat.answer(conv=chat_state,
    #                           img_list=img_list,
    #                           num_beams=args.num_beams,
    #                           temperature=args.temperature,
    #                           max_new_tokens=300,
    #                           max_length=2000)[0]
    #
    # print(llm_message)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="trans")
    parser.add_argument("--mode", type=str, default="part")
    parser.add_argument("--loss", type=str, default="both")
    parser.add_argument("--num-steps", type=int, default=2)

    parser.add_argument("--cfg_path", type=str, default="eval_configs/minigptv2_eval.yaml")
    parser.add_argument("--img_path", type=str, default="")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )


    args = parser.parse_args()
    main(args)