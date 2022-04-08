%%%%%%%%%%%
%% Extract Gabor similarity between images
%%%%%%%%%%

im_dir = "/home/vayzenbe/GitHub_Repos/docnet/stim/original/*.jpg";


im_files = dir(im_dir);

n = 1;
for ii = 1:length(im_files)
    im1 = imread(strcat(im_files(ii).folder,'/', im_files(ii).name));
    im1 = imresize(rgb2gray(im1),[256, 256]);
    
    D{1} = GWTgrid_simple(im1, 0, 0);
    for kk = (ii+1):length(im_files)
        im2 = imread(strcat(im_files(kk).folder,'/', im_files(kk).name));
        im2 = imresize(rgb2gray(im2),[256, 256]);
        D{2} = GWTgrid_simple(im2, 0, 0);
   
        %gbj_rdm(n, 1) = im_files(ii).name(1:end-4);
        %gbj_rdm(n, 2) = im_files(kk).name(1:end-4);
        gbj_rdm(n, 1) = norm(D{1} - D{2});
        n = n +1;
    end
end

csvwrite('rdms/gbj_rdm.csv', gbj_rdm);
%gbj_rdm = array2table(gbj_rdm);
%gbj_rdm.Properties.VariableNames(1:3) = {'obj1', 'obj2', 'similarity'};
%writetable(gbj_rdm, 'gbj_rdm.csv');