# smiless key componenet images
images=(blinkbear/of-watchdog blinkbear/faas-netes blinkbear/openfaas-gateway)
for image in "${images[@]}"; do
    docker pull $image
done



# inferece function images
images=(blinkbear/smiless-speechrecognition blinkbear/smiless-humanactivitypose blinkbear/smiless-translation blinkbear/smiless-imagerecognition blinkbear/smiless-textgeneration blinkbear/smiless-texttospeech blinkbear/smiless-questionanswering blinkbear/smiless-objectdetection blinkbear/smiless-nameentityrecognition blinkbear/smiless-facerecognition blinkbear/smiless-topicmodeling )
for image in "${images[@]}"; do
    docker pull $image
done