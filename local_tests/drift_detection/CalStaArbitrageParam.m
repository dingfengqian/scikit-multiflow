function [Sigmas,val,kStop,msp] = CalStaArbitrageParam(PreData5minCloseA,PreData5minCloseB)

% 基于非参数法计算窗口内时变回归系统
    sig=max(abs(PreData5minCloseB-PreData5minCloseA))/2;
    val=zeros(1,length(PreData5minCloseA));
    for j=1:length(PreData5minCloseA)
        val(j)=fminsearch(@(alpha)sum(exp(-(PreData5minCloseA-PreData5minCloseA(j)).^2/2/sig/sig).*(PreData5minCloseB-alpha*PreData5minCloseA)).^2,1);
    end
    spread = PreData5minCloseB - val'.*PreData5minCloseA;
    msp = mean(PreData5minCloseB - val'.*PreData5minCloseA);
    mspread = spread - msp;
    
    %求去中心化残差的时变标准差
    %Spec=garchset('P',1,'Q',1,'Display','off');
    %[Coeff,~,~,~,Sigmas]=garchfit(Spec,mspread);
    %k_stop=abs(norminv(0.005*ones(length(Sigmas),1),Coeff.C,Sigmas))./Sigmas;

    Spec = garch(1, 1);
    EstMdl = estimate(Spec, mspread, 'Display', 'off');
    disp(EstMdl);
    V = infer(EstMdl,mspread);
    Sigmas = sqrt(V);
    k_stop=abs(norminv(0.005*ones(length(Sigmas),1),EstMdl.Constant,Sigmas))./Sigmas;
    kStop = k_stop(end);
end
