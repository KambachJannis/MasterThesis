function objHeatMap = getHeatMap(path,windows)

%compute the objectness heat map for an image
%INPUT:
%img - input image
%windows - objectness windows (computed using function runObjectness.m)

img = imread(path);
windows = runObjectness(img,windows);

map = zeros(size(img,1),size(img,2));

for idx = 1:size(windows,1)
        xmin = uint16(round(windows(idx,1)));
        ymin = uint16(round(windows(idx,2)));
        xmax = uint16(round(windows(idx,3)));
        ymax = uint16(round(windows(idx,4)));
        score = windows(idx,5);
        maskBox = zeros(size(img,1), size(img,2));
        maskBox(ymin:ymax,xmin:xmax) = score;
        map = map + maskBox;    
end

gray = mat2gray(map);
X = gray2ind(gray,256);
Y = ind2rgb(X,jet(256));
objHeatMap = gray;
figure;
subplot(2,1,1),imshow(img),title('Input image');
subplot(2,1,2),imshow(objHeatMap);title('Objectness heat map');