clear all
%% Loding of matrices and vectors
R=readmatrix('R.txt');
T=readmatrix('T.txt');
T=T(3:5,1);
K1=readmatrix('Kl.txt');
K2=readmatrix('Kr.txt');

%%Creating the P/A-matrices

K1=[K1, [0, 0, 1]'];
K2=[K2, [0, 0, 1]'];

RT=[R, T];
RT=[RT; [0,0,0,1]];

P1=K1*eye(4,4);
P2=K2*RT;

%%Calculating the points

for t=1:1:1
 
    %fileID2 =fopen(strcat('pws/000POS',num2str(t),'Rekonstruktion',num2str(30),'.pcf'));
    %fileID2 =fopen(strcat('Reconstruction_of_Bust_of_Friedrich_Schiller.pcf'));
    fileID2 =fopen(strcat('Maus1.pcf'));
    dump2 = fscanf(fileID2, '%s', [1 1]);               %Schmeißt erste Zeilen weg
    dump2 = fscanf(fileID2, '%s %s %s %s %s', [5 1]);
    coord2 = fscanf(fileID2, '%d %d %f %f %f %f %f %f %f %f %f', [11 inf]); %Ordnet spalten 
    coord2 = coord2';    
    %coord2(coord2(:,5)<0.6,:)=[];                    %Korrelationskoeffezientengrenzwert
    %coord2(coord2(:,8)>2.5,:)=[]; 
    Liste=zeros(size(coord2,1),14);
    Liste(:,1:4)=coord2(:,1:4);
    % Liste(:,9:11)=coord2(:,9:11);
    Liste(:,12:14)=coord2(:,6:8);
   for index=1:1:size(coord2,1)

    x1=Liste(index,1);
    y1=Liste(index,2);
    x2=Liste(index,3);
    y2=Liste(index,4);
%% Setting up the LESS
    LES1=x1*P1(3,:)-P1(1,:);
    LES2=y1*P1(3,:)-P1(2,:);
    LES3=x2*P2(3,:)-P2(1,:);
    LES4=y2*P2(3,:)-P2(2,:);

    LES=[LES1;LES2;LES3;LES4];
%% Solving the LES
    [~,~,V] = svd(LES);
    S = V(:,end);
    S = S ./ S(4);


    Liste(index,6)=S(1); 
    Liste(index,7)=S(2);
    Liste(index,8)=S(3); 
   end

end
%% Creating point clouds
% Liste=Liste';
Liste(Liste(:,8)==0,:)=[];
% Liste=sortrows(Liste);
t=1;
D1=Liste(:,6:11);
sx=sprintf('pcs/Maus%.1d.xyz',t);
dlmwrite(sx,D1);
sy=sprintf('pcs/Maus%.1d.pcf',t);
dlmwrite(sy,Liste);