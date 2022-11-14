clear all
close all

% Training neutral faces 
eig_vec_neutral = face_recognition(1,0);
% training Smile faces
eig_vec_smile = face_recognition(2,0);

%Reconstruction of neutral face from training PC
face_recognition(3,eig_vec_neutral);
%reconstruction of smile face from training PC
face_recognition(4,eig_vec_smile);

face_recognition(5,eig_vec_neutral);
face_recognition(6,eig_vec_smile);

%face classification
EIG=[eig_vec_neutral eig_vec_smile];

face_recognition(7,EIG);

