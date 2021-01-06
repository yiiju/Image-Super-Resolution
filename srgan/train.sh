python srgan.py --n_epochs 100 \
                --batch_size 8 \
                --hr_height 400 \
                --hr_width 400 \
                --gpu_ids 8 \
                --dataset_name train_resize_sq \
                --checkpoint_interval 1 \
                --save_folder 'save400'