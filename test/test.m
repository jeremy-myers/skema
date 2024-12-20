clear;

model = "Sparse";
model_str = "sparsesign";
fbasename = "rectangular_dense";
fext = "txt";
model_ext = "mtx";
matrix_dir = "./data";

fmatrix = sprintf("%s/%s.%s", matrix_dir, fbasename, fext);

if strcmp(fext, "mtx")
    A = mmread(fmatrix);
elseif strcmp(fext, "txt")
    A = readmatrix(fmatrix, "Delimiter", " ");
else
    fprintf("Uh oh!");
end

[m,n] = size(A);

r = 3;
k = 5;
s = 7;

sketch = ThreeSketch(model, m, n, k, s);

output_dir = ".";

fUpsilon = sprintf("%s/%s_%s_Upsilon.%s", output_dir, fbasename, model_str, model_ext);
fOmega = sprintf("%s/%s_%s_Omega.%s", output_dir,fbasename, model_str, model_ext);
fPhi = sprintf("%s/%s_%s_Phi.%s", output_dir,fbasename, model_str, model_ext);
fPsi = sprintf("%s/%s_%s_Psi.%s", output_dir,fbasename, model_str, model_ext);

fX = sprintf("%s/%s_%s_X.txt", output_dir,fbasename, model_str);
fY = sprintf("%s/%s_%s_Y.txt", output_dir,fbasename, model_str);
fZ = sprintf("%s/%s_%s_Z.txt", output_dir,fbasename, model_str);

fU = sprintf("%s/%s_%s_U.txt", output_dir,fbasename, model_str);
fS = sprintf("%s/%s_%s_S.txt", output_dir,fbasename, model_str);
fV = sprintf("%s/%s_%s_V.txt", output_dir,fbasename, model_str);

fR = sprintf("%s/%s_rnrms.txt", output_dir,fbasename);

if strcmp(model_ext, "txt")
    Upsilon = readmatrix(fUpsilon, "Delimiter", " ");
    Omega = readmatrix(fOmega, "Delimiter", " ");
    Phi = readmatrix(fPhi, "Delimiter", " ");
    Psi = readmatrix(fPsi, "Delimiter", " ");
elseif strcmp(model_ext, "mtx")
    Upsilon = mmread(fUpsilon);
    Omega = mmread(fOmega);
    Phi = mmread(fPhi);
    Psi = mmread(fPsi);
else
    fprintf("Uh oh2");
end
   

X = readmatrix(fX, "Delimiter", " ");
Y = readmatrix(fY, "Delimiter", " ");
Z = readmatrix(fZ, "Delimiter", " ");

U = readmatrix(fU, "Delimiter", " ");
S = readmatrix(fS, "Delimiter", " ");
V = readmatrix(fV, "Delimiter", " ");

sketch.Upsilon = Upsilon;
sketch.Omega = Omega;
sketch.Phi = Phi;
sketch.Psi = Psi;

sketch.LinearUpdate(A);

fprintf("Max diff X: %.16f\n", max(max(abs(sketch.X) - abs(X))));
fprintf("Max diff Y: %.16f\n", max(max(abs(sketch.Y) - abs(Y))));
fprintf("Max diff Z: %.16f\n", max(max(abs(sketch.Z) - abs(Z))));

[u,s,v] = sketch.FixedRankApprox(r);

fprintf("Max diff U: %.16f\n", max(max(abs(u) - abs(U))));
fprintf("Max diff S: %.16f\n", max(max(abs(s) - abs(S))));
fprintf("Max diff V: %.16f\n", max(max(abs(v) - abs(V))));

R = readmatrix(fR, "Delimiter", " ");

s = diag(s);
rr = zeros(r,1);
for i = 1:r
    rr(i) = sqrt(norm([A'*u(:,i)-s(i)*v(:,i); A*v(:,i)-s(i)*u(:,i)]));
end

RR = zeros(r,1);
for i = 1:r
    RR(i) = sqrt(norm([A'*U(:,i)-S(i)*V(:,i); A*V(:,i)-S(i)*U(:,i)]));
end

