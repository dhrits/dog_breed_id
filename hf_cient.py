from gradio_client import Client, handle_file
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Simple client for dog breed identification API',
        description='Given an image path, identifies the dog breed',
    )
    parser.add_argument('-i', '--impath', help='path or url to input image with dog breed to identify', required=True)
    args = parser.parse_args()

    client = Client("deman539/dog_breed_id")
    annotated_impath, breed, confidence = client.predict(
        img=handle_file(args.impath),
        api_name="/predict"
    )
    print(f'Annotated image: {annotated_impath}')
    print(f'Breed: {breed["label"]}, confidence: {confidence*100:.2f}%')
