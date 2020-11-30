# from cx1g8 to fias:
# copy everything but the experiment dir
# rsync -anv --delete --exclude=experiments/* --exclude=.git/* ~/Code/coppelia_sim_inverse_model/ wilmot@fias.uni-frankfurt.de:~/Documents/code/coppelia_sim_inverse_model

# echo    # (optional) move to a new line
# echo    # (optional) move to a new line
# read -p "Do you want to proceed? " -n 1 -r
# echo    # (optional) move to a new line
# if [[ ! $REPLY =~ ^[Yy]$ ]]
# then
#     exit 1
# fi
rsync -a --delete --exclude=experiments/* --exclude=.git/* ~/Code/coppelia_sim_inverse_model/ wilmot@fias.uni-frankfurt.de:~/Documents/code/coppelia_sim_inverse_model
