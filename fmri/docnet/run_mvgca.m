clear all;
%%
%set up dirs and sub/roi params
study_dir = '/lab_data/behrmannlab/vlad/docnet';
out_dir = '/user_data/vayzenbe/GitHub_Repos/docnet/results/mvgca';
addpath('/user_data/vayzenbe/GitHub_Repos/MVGC1')
startup;

subj_list = [2001,2002,2003,2004, 2005, 2007, 2008, 2012, 2013, 2014, 2015, 2016, 2017, 2018];
subj_list = [2001,2002,2003,2004, 2005, 2007, 2008, 2012, 2013, 2014, 2015, 2016];


lr = {'l','r'};
dorsal_rois = {'PPC_spaceloc',   'APC_spaceloc',  'APC_distloc'};
ventral_rois = {'LO_toolloc'};
cols = {'sub'}; %
file_suf = ''
%%
%start analysis loop
sn = 1; %tracks which sub num we are on
for ss = subj_list
   results_dir = [study_dir,'/sub-docnet',int2str(ss),'/ses-02/','/derivatives/results/beta_ts'];
   sub_summary{sn, 1} = ss;
   rn = 2; %rn starts at 2 because col 1 is sub
   for dlr = {'l','r'}
      for  drr = dorsal_rois
         
         droi = [dlr{1},drr{1}];
         dorsal_file = [results_dir, '/',dlr{1},drr{1},'_pc_ts.mat'];
         %check if dorsal roi exists
         if exist(dorsal_file)
            dorsal_ts = cell2mat(struct2cell(load(dorsal_file))); %load .mat file and convert to mat
            dorsal_times = size(dorsal_ts); %save size for later
         else
             rn = rn +1;
             continue
         end

         for vlr = {'l','r'}
             for vrr = ventral_rois
                vroi = [vlr{1},vrr{1}];

                
                %for first sub add rois to col cell to eventually make the
                %summary columns
                if ss == 2001
                    cols{end+1} = [droi,'_',vroi];
                end
                ventral_file = [results_dir, '/',vlr{1},vrr{1},'_pc_ts.mat']
                
                if exist(ventral_file)
                    ventral_ts = cell2mat(struct2cell(load([results_dir, '/',vlr{1},vrr{1},'_pc_ts.mat'])));
                    ventral_times = size(ventral_ts);
                    
                    %determine what the min number of PCs to use
                    %mvgca has to have same number of TSs
                    pc_n = min([dorsal_times(3),ventral_times(3)]); 
                    
                    %setup empty 3D tensor and add dorsal and ventral TSs
                    X = zeros(2, dorsal_times(2),pc_n);
                    X(1,:,:)= dorsal_ts(:,:,1:pc_n);
                    X(2,:,:)= ventral_ts(:,:,1:pc_n);

                    %run mvgca
                    [F, p] = mvgc_ts(X);
                    f_diff = F(2,1) - F(1,2); %calculate f-diff by subtracting region predictors from eachother
                    
                    %add diff to cell where sn is the sub row and rn is the
                    %roi col
                    sub_summary{sn, rn} = f_diff;
                    rn = rn +1;
                else
                    rn = rn+1;
                    break
                end
             end
         end
      end

   end
   
   sn = sn + 1;
end
%%
%convert final summary to table and save
final_summary = cell2table(sub_summary, 'VariableNames', cols);
writetable(final_summary, [out_dir,'/mvgca_summary', file_suf,'.csv'], 'Delimiter', ',')
