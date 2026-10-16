clear all

close all


N=100; %% numero de regiones 

Tau=1;  %define your time shift
sigma=0.01;

epsFC=0.0002;
epsFCtau=0.00004;
maxC=0.1;

Isubdiag = find(tril(ones(N),-1));


sigma=0.01;
f_diff=0.025*ones(N,1);
% ceff

for nsub=1:NSUB
    nsub
    % load timeseries is missing

    FCemp=corrcoef(ts');
    COVemp=cov(ts');
    % COV(tau)
    tst=ts';
    for i=1:N
        for j=1:N
            sigratio(i,j)=1/sqrt(COVemp(i,i))/sqrt(COVemp(j,j));
            [clag lags] = xcov(tst(:,i),tst(:,j),Tau);
            indx=find(lags==Tau);
            COVtauemp(i,j)=clag(indx)/size(tst,1);
        end
    end
    COVtauemp=COVtauemp.*sigratio;





    Cnew=C;
    olderror=100000;
    for iter=1:5000
        % Linear Hopf FC
        [FCsim,COVsim,aux, A]=hopf_int(Cnew,f_diff,sigma);
        COVtausim=expm((Tau*TR)*A)*COVsim;
        COVtausim=COVtausim(1:N,1:N);
        for i=1:N
            for j=1:N
                sigratiosim(i,j)=1/sqrt(COVsim(i,i))/sqrt(COVsim(j,j));
            end
        end
        COVtausim=COVtausim.*sigratiosim;
        errorFC(iter)=mean(mean((FCemp-FCsim).^2));
        errorCOVtau(iter)=mean(mean((COVtauemp-COVtausim).^2));

        if mod(iter,100)<0.1
            errornow=mean(mean((FCemp-FCsim).^2))+mean(mean((COVtauemp-COVtausim).^2));
            if  (olderror-errornow)/errornow<0.01
                break;
            end
            if  olderror<errornow
                break;
            end
            olderror=errornow;
        end

        for i=1:N
            for j=1:N
                if (C(i,j)>0 || j==N/2+i)
                    Cnew(i,j)=Cnew(i,j)+epsFC*(FCemp(i,j)-FCsim(i,j)) ...
                        +epsFCtau*(COVtauemp(i,j)-COVtausim(i,j));
                    if Cnew(i,j)<0
                        Cnew(i,j)=0;
                    end
                end
            end
        end
        Cnew = Cnew/max(max(Cnew))*maxC;
    end
    Ceff=Cnew;
    Ceff(nsub,:,:)=Ceff;

% final iteration
    [FCsim,COVsim,aux, A]=hopf_int(Cnew,f_diff,sigma);
    fittFC(nsub)=corr2(FCemp(Isubdiag),FCsim(Isubdiag));
    COVtausim=expm((Tau*TR)*A)*COVsim;
    COVtausim=COVtausim(1:N,1:N);
    for i=1:N
        for j=1:N
            sigratiosim(i,j)=1/sqrt(COVsim(i,i))/sqrt(COVsim(j,j));
        end
    end
    COVtausim=COVtausim.*sigratiosim;
    fittCVtau(nsub)=corr2(COVtauemp(Isubdiag),COVtausim(Isubdiag));
end


