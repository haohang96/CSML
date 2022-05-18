python main_lincls.py \
--nomoxing \
--train_url=../single_crop_200ep \
--data_dir=/mnt/imagenet2012 \
--usupv_lr=0.03 \
--usupv_batch=256 \
--pretrained_epoch=199 \
--init_lr=30. \
--batch_size=256 \
--wd=0. \
--selected_feat_id=17
#--resume=true \
