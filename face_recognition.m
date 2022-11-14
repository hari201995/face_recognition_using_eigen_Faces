function [normalized_eig_vec_C]=face_recognition(type,principal_cmp)


%% get the N^2x1 vector
% names = ['2a','2b','6a','6b','7','7','9','9','10','10','11','11','12','12','13','13','15','15','17','17','18','18','19','19','20','20','21','21'...
%         '22','22','23','23','24','24'];

% d = dir('C:\Users\mailh\OneDrive\Documents\MATLAB\face recognition\eigenface_dataset\upload_dataset\*.jpg');
% names = {d.name};
% for i=1:2:length(names)
%     cd 'C:\Users\mailh\OneDrive\Documents\MATLAB\face recognition\eigenface_dataset\upload_dataset';
%     img= double(imread(names{i}));
%     [row,col] = size(img);
%     img_vec(:,i) = reshape(img,row*col,1);
% end

image='C:\Users\mailh\OneDrive\Documents\MATLAB\face recognition\eigenface_dataset\upload_dataset';
file=dir(fullfile('C:\Users\mailh\OneDrive\Documents\MATLAB\face recognition\eigenface_dataset\upload_dataset','*.jpg'));
images=numel(file);
final={};

for n=1:images
    f=fullfile(image,file(n).name);
    our_images1=imread(f);
    our_images=im2double(our_images1);
    our_images_reshape=reshape(our_images,[193*162 1]);
    final=[final our_images_reshape];
end
final = cell2mat(final);
neutral = final(:,1:2:end);
smile = final(:,2:2:end);

N = neutral(:,1:100);
avg_img = sum(N,2)/100;
N_test_1 = neutral(:,171);
N_test_2 = smile(:,171);
%% Neutral Image
%% Qn1

if type==1

%% compute the average
% avg_img = 2*sum(img_vec,2)/length(names);
N = neutral(:,1:100);
avg_img = sum(N,2)/100;
%% subtract the mean face
% A = zeros(row*col,length(names)*0.5);
for i=1:100
    phi(:,i) = N(:,i) -(avg_img); 
end

%% compute the co variance matrix
% C = (A*A')/length(names);
%% compute the eigen vectors of A^t*A
modified_C = (phi'*phi);
[eig_vec,eig_val] = eig(modified_C); 

figure;
plot(diag(eig_val)./sum(diag(eig_val)))
title('normalized eigen values');

%% compute the eigen vectors of u(i) A*A^t and normalize it. 
eig_vec_C = phi*eig_vec;
normalized_eig_vec_C = normalize(eig_vec_C,'norm');
 for i=100:-1:90
        reshaped_eig_vec= reshape(normalized_eig_vec_C(:,i),[193 162]);
        figure;
        imagesc(reshaped_eig_vec);
        colormap("gray")
 end
end

%% smiling face
%% Qn2
if type==2

%% compute the average
% avg_img = 2*sum(img_vec,2)/length(names);
N = smile(:,1:100);
avg_img = sum(N,2)/100;
%% subtract the mean face
% A = zeros(row*col,length(names)*0.5);
for i=1:100
    phi(:,i) = N(:,i) -(avg_img); 
end

%% compute the co variance matrix
% C = (A*A')/length(names);
%% compute the eigen vectors of A^t*A
modified_C = (phi'*phi);
[eig_vec,eig_val] = eig(modified_C); 

plot(diag(eig_val)./sum(diag(eig_val)))

%% compute the eigen vectors of u(i) A*A^t and normalize it. 
eig_vec_C = phi*eig_vec;
normalized_eig_vec_C = normalize(eig_vec_C,'norm');

    %% plot the eigen faces
    for i=100:-1:90
        reshaped_eig_vec = reshape(normalized_eig_vec_C(:,i),[193 162]);
        figure;
        imagesc(reshaped_eig_vec);
        colormap("gray")
    end
end

%% reconstruction using PC's 
if type==3
    N = neutral(:,1:100);
    avg_img = sum(N,2)/100;
    %% subtract the mean face
    % A = zeros(row*col,length(names)*0.5);
    for i=1:100
        phi(:,i) = N(:,i) -(avg_img); 
    end
    MSE=[];
    coeff = transpose(principal_cmp)*phi;
        for ii=100:-1:1
            reconstructed_img = principal_cmp(:,100:-1:ii)*coeff(100:-1:ii, 1);
            reconstructed_img = reconstructed_img + avg_img;
            reshaped_recon_img = reshape(reconstructed_img(:,1),[193 162]);
    %         figure;
    %         imagesc(reshaped_recon_img);
    %         colormap("gray")
    
            %% MSE computation
            mse= norm(reconstructed_img-phi(:,1),2);
            MSE=[MSE mse];
        end
        figure;
        plot(MSE)
        title('Variation of reconstruction of image 121 not a part of training set wrt number of Eigen values taken for reconstruction');
        xlabel('number of eigen values taken');
        ylabel('MSE between reconstructed image and Natural image');
end


if type==4
    N = smile(:,1:100);
    avg_img = sum(N,2)/100;
    %% subtract the mean face
    % A = zeros(row*col,length(names)*0.5);
    for i=1:100
        phi(:,i) = N(:,i) -(avg_img); 
    end
    MSE=[];
    coeff = transpose(principal_cmp)*phi;
        for ii=100:-1:1
            reconstructed_img = principal_cmp(:,100:-1:ii)*coeff(100:-1:ii, 1);
            reconstructed_img = reconstructed_img + avg_img;
            reshaped_recon_img = reshape(reconstructed_img(:,1),[193 162]);
    %         figure;
    %         imagesc(reshaped_recon_img);
    %         colormap("gray")
    
            %% MSE computation
            mse= norm(reconstructed_img-phi(:,1),2);
            MSE=[MSE mse];
        end
        figure;
        plot(MSE)
        title('Variation of reconstruction of image 121 not a part of training set wrt number of Eigen values taken for reconstruction');
        xlabel('number of eigen values taken');
        ylabel('MSE between reconstructed image and Natural image');
end

%% reconstruction using PC's 
if type==5
    N = N_test_1;
    %% subtract the mean facts
    phi=N;
    MSE=[];
    coeff = transpose(principal_cmp)*phi;
        for ii=100:-1:1
            reconstructed_img = principal_cmp(:,100:-1:ii)*coeff(100:-1:ii, 1);
            reconstructed_img = reconstructed_img + avg_img;
            reshaped_recon_img = reshape(reconstructed_img(:,1),[193 162]);
    %         figure;
    %         imagesc(reshaped_recon_img);
    %         colormap("gray")
    
            %% MSE computation
            mse= norm(reconstructed_img-phi(:,1),2);
            MSE=[MSE mse];
        end
    figure;
    plot(MSE)
    title('Variation of reconstruction of image 121 not a part of training set wrt number of Eigen values taken for reconstruction');
    xlabel('number of eigen values taken');
    ylabel('MSE between reconstructed image and Natural image');
end

%% reconstruction using PC's of other 1 among 71 images
if type==6
    N = smile(:,171);
    phi = N; 
    MSE=[];
    coeff = transpose(principal_cmp)*phi;
    for ii=100:-1:1
        coeff = transpose(principal_cmp(:,ii:100))*phi;
        reconstructed_img = principal_cmp(:,ii:100)*coeff;
        reconstructed_img = reconstructed_img + avg_img;
        reshaped_recon_img = reshape(reconstructed_img(:,1),[193 162]);
%         figure;
%         imagesc(reshaped_recon_img);
%         colormap("gray")

        %% MSE computation
        mse= norm(reconstructed_img(:,1)-phi,2);
        MSE=[MSE mse];
    end
    figure;
    plot(MSE)
    title('Variation of reconstruction of image 121 not a part of training set wrt number of Eigen values taken for reconstruction');
    xlabel('number of eigen values taken');
    ylabel('MSE between reconstructed image and Natural image');
end

%% face classification using Eigen faces.

if type==7

    smile_eig_vec = principal_cmp(:,1:100);
    neutral_eig_vec = principal_cmp(:,101:200);

    test_vector = [neutral(:,end-30:end) smile(:,end-30:end)];

    avg_img = sum(test_vector,2)/100;
        phi = test_vector -(avg_img); 
    test_input =[ones(1,30) zeros(1,30)];

    coeff_neutral = transpose(neutral_eig_vec)*phi;
    coeff_smile = transpose(smile_eig_vec)*phi;

    for im=1:60
        N= test_vector(:,im);
        reconstructed_img_neutral = smile_eig_vec(:,100:80)*coeff_neutral(100:80,im);
        reconstructed_img_neutral = reconstructed_img_neutral + avg_img;
        reshaped_recon_img_neutral = reshape(reconstructed_img_neutral(:,1),[193 162]);
        mse_neutral(im) = norm(reconstructed_img_neutral-N,2);
    end

    for im=1:60
        N= test_vector(:,im);
        reconstructed_img_smile = smile_eig_vec(:,100:80)*coeff_smile(100:80,im);
        reconstructed_img_smile = reconstructed_img_smile + avg_img;
        reshaped_recon_img_smile = reshape(reconstructed_img_smile(:,1),[193 162]);
        mse_smile(im) = norm(reconstructed_img_smile(:,1)-N,2);
    end

    for im=1:60
        if mse_neutral(im)<=mse_smile(im)
            test_output(im)=1;
        else
            test_output(im)=0;
        end
    end
   
    comp_vec = test_output == test_input;
    success_percent = sum(comp_vec)/60; 
 end
end





