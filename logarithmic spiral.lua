--    (c) 2016, 2017, 2018 Navid Kardan 
--    This program is free software: you can redistribute it and/or modify
--    it under the terms of the GNU General Public License as published by
--    the Free Software Foundation, either version 3 of the License, or
--    (at your option) any later version.
--
--    This program is distributed in the hope that it will be useful,
--    but WITHOUT ANY WARRANTY; without even the implied warranty of
--    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
--    GNU General Public License for more details.
--
--    You should have received a copy of the GNU General Public License
--    along with this program.  If not, see <http://www.gnu.org/licenses/>.




--This code doesn't use GPU. Running time is about an hour on a descent system
require 'torch'
require 'nn'
require 'cunn'
require 'paths'
require'image'
require'optim'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

ovc=torch.Tensor({50,50})

function ind2label(ind,ovcs)
  local tmp=0
  for i=1,ovcs:size(1) do
    tmp=tmp+ovcs[i]
    if ind<=tmp then
      return i-1
    end
  end
end

function transfrom_labels2(labelinds,numclass,ovcs) 
local sz=labelinds:size()[1]
local ovc=torch.sum(ovcs)
local newlabs=torch.Tensor(sz,ovc):zero();
  for i=1,sz do
    for j=1,ovc do
      local tmp=ind2label(j,ovcs)
      if labelinds[i]== tmp then
        newlabs[{{i},{j}}]=1/ovcs[tmp+1]
      end
    end
  end
  return newlabs
end

function transfrom_predict2(predict,numclass,ovcs)
  local sz=predict:size()[1]

  local tmppredict=torch.Tensor(sz,numclass):fill(1) 
  local offset=0
  for j=1,numclass do
    for k=1,ovcs[j] do
      tmppredict[{{},{j}}]:cmul(ovcs[j]*predict[{{},{k+offset}}])

    end
    offset=offset+ovcs[j]
  end
  return tmppredict
end

function create_ds()
--logarithmic spiral
  local a=0.3;
  local b=0.1;
  local r=.3;
  local xp = torch.linspace(0,4*math.pi,500)
 
  local y1 = a*torch.cmul(torch.sin(xp) , torch.exp(b*xp))
  local y2 = a*torch.cmul(torch.cos(xp) , torch.exp(b*xp))
  d1=torch.cat(y1,y2,2)
  c1=torch.zeros(xp:size(1))  
  local x=0;
  localy=0;
  local r=1.1;
  local xp = torch.linspace(0,2*math.pi,500)
  local y1 = torch.sin(xp)*r
  local y2 = torch.cos(xp)*r
  d2=torch.cat(y1,y2,2)
  c2=torch.ones(xp:size(1))    
  return {torch.cat(d1,d2,1),torch.cat(c1,c2,1)}
end


function create_mininet(inps)
  local net=nn.Sequential()
  last=nn.Linear(inps,1)
  net:add(last)
  fcinit(net)
  return net
end

function create_net(inps,outs)
  local net=nn.Sequential()
  net:add(nn.Linear(inps,400))
  net:add(nn.BatchNormalization(400))
  net:add(nn.ReLU()) 
  net:add(nn.Linear(400,400))
  net:add(nn.BatchNormalization(400))
  net:add(nn.ReLU())  
  tree=nn.Linear(400,outs)
  fcinit(tree)
  net:add(tree)
  return net
end



function train(net,tr,cl,lr)
  net:zeroGradParameters()
  local sf=nn.SoftMax()
  local netouts=sf:forward(net:forward(tr))
  local err=(netouts-cl) *(1/tr:size(1))
  net:backward(tr,err)
  net:updateParameters(lr);
end

function fcinit(model)
   for k,v in pairs(model:findModules'nn.Linear') do
     v.weight:normal(0,1)
     if v.bias ~=nil then
       v.bias:normal(0,1)
     end
   end
end



ds=create_ds()
net=create_net(2,torch.sum(ovc))
params, gradParams = net:getParameters()

lr=0.1 --0.02
minibt=1000
labels=transfrom_labels2(ds[2],2,ovc)
for i=1,5000 do
  
  for j=1,1000/minibt do
    train(net,ds[1][{{(j-1)*minibt+1,j*minibt},{}}],labels[{{(j-1)*minibt+1,j*minibt}}],lr)
  end
end
print('network arch.' .. net:__tostring())
net:evaluate()

for x=-4,4,0.01 do
  ind=1
  y=torch.linspace(-4,4,800):view(800,1)
  xrep=torch.Tensor(800,1):fill(x)
  tten=torch.cat(xrep,y)
  local t=transfrom_predict2(nn.SoftMax():forward(net:forward(tten)), 2 ,ovc)
  if x==-4 then
    activationmap1=t[{{},1}]
    activationmap2=t[{{},2}]
  else
    activationmap1=torch.cat(activationmap1,t[{{},1}],2)
    activationmap2=torch.cat(activationmap2,t[{{},2}],2)
  end
end
image.save('spiral_a.png',activationmap1)
image.save('circle_b.png',activationmap2)

