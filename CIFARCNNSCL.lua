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

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'paths'
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)


function ind2label(ind,ovcs)
  local tmp=0
  for i=1,ovcs:size(1) do
    tmp=tmp+ovcs[i]
    if ind<=tmp then
      return i-1
    end
  end
end

function transfrom_labels2(labelinds,numclass,ovcs) --separate ovc for each class, assuming class indices start from zero!!!
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
  local tmppredict=torch.Tensor(sz,numclass):fill(1):cuda() 
  local offset=0
  for j=1,numclass do
    for k=1,ovcs[j] do
      tmppredict[{{},{j}}]:cmul(ovcs[j]*predict[{{},{k+offset}}])
    end
    offset=offset+ovcs[j]
  end
  return tmppredict
end

function transfrom_predict2sum(predict,numclass,ovcs)
  local sz=predict:size()[1]

  local tmppredict=torch.Tensor(sz,numclass):fill(0):cuda() 
  local offset=0
  for j=1,numclass do
    for k=1,ovcs[j] do
      tmppredict[{{},{j}}]:add(predict[{{},{k+offset}}])
    end
    offset=offset+ovcs[j]
  end
  return tmppredict
end

function create_ds_mnist()

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

if not paths.dirp('mnist.t7') then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

train_file = 'mnist.t7/train_32x32.t7'
test_file = 'mnist.t7/test_32x32.t7'
trainData = torch.load(train_file,'ascii')
testData = torch.load(test_file,'ascii')


  return {trainData.data:float():cuda()/256,trainData.labels:type('torch.CudaLongTensor')-1,testData.data:float():cuda()/256,testData.labels:type('torch.CudaLongTensor')-1} 
end

function create_ds_cifar10()
  if (not paths.filep("cifar10torch.zip")) then
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torch.zip')
    os.execute('unzip cifar10torch.zip')
  end
  trainset = torch.load('cifar10-train.t7')
  testset = torch.load('cifar10-test.t7')

  local t=trainset.data[{{1,50000},{},{}}]:float():cuda()
  local tc=trainset.label[{{1,50000}}]:type('torch.CudaLongTensor') 
  local tt=testset.data[{{1,10000},{},{}}]:float():cuda()
  local ttc=testset.label[{{1,10000}}]:type('torch.CudaLongTensor')     

  return {t/256,tc,tt/256,ttc}
end


local function w_init_kaiming(fan_in, fan_out)
   return math.sqrt(4/(fan_in + fan_out))
end
-- "Efficient backprop"
-- Yann Lecun, 1998
local function w_init_heuristic(fan_in, fan_out)
   return math.sqrt(1/(3*fan_in))
end


-- "Understanding the difficulty of training deep feedforward neural networks"
-- Xavier Glorot, 2010
local function w_init_xavier(fan_in, fan_out)
   return math.sqrt(2/(fan_in + fan_out))
end


-- "Understanding the difficulty of training deep feedforward neural networks"
-- Xavier Glorot, 2010
local function w_init_xavier_caffe(fan_in, fan_out)
   return math.sqrt(1/fan_in)
end


-- "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
-- Kaiming He, 2015
local function w_init_kaiming(fan_in, fan_out)
   return math.sqrt(4/(fan_in + fan_out))
end

local function w_init(net, arg)
   -- choose initialization method
   local method = nil
   if     arg == 'heuristic'    then method = w_init_heuristic
   elseif arg == 'xavier'       then method = w_init_xavier
   elseif arg == 'xavier_caffe' then method = w_init_xavier_caffe
   elseif arg == 'kaiming'      then method = w_init_kaiming
   else
      assert(false)
   end

   -- loop over all convolutional modules
   for i = 1, #net.modules do
      local m = net.modules[i]
      if m.__typename == 'nn.SpatialConvolution' then
         m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
      elseif m.__typename == 'nn.SpatialConvolutionMM' then
         m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
      elseif m.__typename == 'cudnn.SpatialConvolution' then
         m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
      elseif m.__typename == 'nn.LateralConvolution' then
         m:reset(method(m.nInputPlane*1*1, m.nOutputPlane*1*1))
      elseif m.__typename == 'nn.VerticalConvolution' then
         m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW))
      elseif m.__typename == 'nn.HorizontalConvolution' then
         m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW))
      elseif m.__typename == 'nn.Linear' then
         m:reset(method(m.weight:size(2), m.weight:size(1)))
      elseif m.__typename == 'nn.TemporalConvolution' then
         m:reset(method(m.weight:size(2), m.weight:size(1)))            
      end

      if m.bias then
         m.bias:zero()
      end
   end
   return net
end


function create_mininet(inps)
  local net=nn.Linear(inps,1)
  return net
end


function create_net(inps,outs)
  local net=nn.Sequential()
  net:add(nn.SpatialConvolution(3,128,3,3,1,1,1,1)) 
  net:add(nn.SpatialBatchNormalization(128))
  net:add(nn.ReLU())
  net:add(nn.SpatialMaxPooling(2,2))
  net:add(nn.SpatialConvolution(128,256,3,3,1,1,1,1))
  net:add(nn.SpatialBatchNormalization(256))
  net:add(nn.ReLU())
  net:add(nn.SpatialMaxPooling(2,2)) 
  net:add(nn.SpatialConvolution(256,512,3,3,1,1,1,1))
  net:add(nn.SpatialBatchNormalization(512))
  net:add(nn.ReLU())
  net:add(nn.SpatialMaxPooling(2,2))
  net:add(nn.SpatialConvolution(512,512,3,3,1,1,1,1))
  net:add(nn.SpatialBatchNormalization(512))
  net:add(nn.ReLU())
  net:add(nn.SpatialMaxPooling(2,2))
  net:add(nn.SpatialConvolution(512,1024,2,2))
  net:add(nn.SpatialBatchNormalization(1024))

  net:add(nn.ReLU())
  net:add(nn.View(1024))
  tree=nn.Concat(2);
  for i=1,outs do
    tree:add(create_mininet(1024))
  end
  net:add(tree)
  return net
end


function train(net,tr,cl,lr)
  net:zeroGradParameters()
  local sf=nn.SoftMax():cuda()
  local netouts=sf:forward(net:forward(tr))
  local err=(netouts-cl) *(1/tr:size(1))    
  net:backward(tr,err)
  net:updateParameters(lr);
end


------------------------------------------------------------Defining the subsets and the omega for each aggregate (10). To train regular NN just set mega to one (uncomment the ovc line) 

subsets={{1,2},{3,4},{5,6},{7,8},{9,10}}
ovc=torch.Tensor({{10,10},{10,10},{10,10},{10,10},{10,10}})
--ovc=torch.Tensor({{1,1},{1,1},{1,1},{1,1},{1,1}})

ds=create_ds_cifar10()  
numberofclasses=10
inputchannels=3

-------------------------------------------------------


ds[5]=ds[1][{{1,torch.floor(ds[1]:size(1)/10)},{},{},{}}]:clone() -----validation set data
ds[1]=ds[1][{{torch.floor(ds[1]:size(1)/10),ds[1]:size(1)},{},{},{}}]:clone()
ds[6]=ds[2][{{1,torch.floor(ds[2]:size(1)/10)}}]:clone() -----validation set labels
ds[2]=ds[2][{{torch.floor(ds[2]:size(1)/10),ds[2]:size(1)}}]:clone()
n={}
dspart={}

for i=1,numberofclasses do
  n[i]=torch.sum(torch.eq(ds[2],i-1))
  dspart[i]=torch.Tensor(n[i],inputchannels,32,32):cuda()
end
counter=torch.ones(10)
for i=1,ds[1]:size(1) do
  dspart[ds[2][i]+1][counter[ds[2][i]+1] ]=ds[1][i]:clone()
  counter[ds[2][i]+1]=counter[ds[2][i]+1]+1
end

valdspart={}
for i=1,numberofclasses do
  n[i]=torch.sum(torch.eq(ds[6],i-1))
  valdspart[i]=torch.Tensor(n[i],inputchannels,32,32):cuda()
end
counter=torch.ones(10)
for i=1,ds[5]:size(1) do
  valdspart[ds[6][i]+1][counter[ds[6][i]+1] ]=ds[5][i]:clone()
  counter[ds[6][i]+1]=counter[ds[6][i]+1]+1
end

dses={}
dsescl={}
dseslabels={}
valdses={}
valdsescl={}
nets={}
optimalnets={}
optimalnets2={}
optimalaccs={}
optimalaccs2={}
for i=1,#subsets do
  dses[i]=dspart[subsets[i][1] ]:clone()
  dsescl[i]=torch.zeros(dspart[subsets[i][1] ]:size(1))
  valdses[i]=valdspart[subsets[i][1] ]:clone()
  valdsescl[i]=torch.zeros(valdspart[subsets[i][1] ]:size(1))
  for j=2,#subsets[i] do
    dses[i]=torch.cat(dses[i],dspart[subsets[i][j] ],1)
    dsescl[i]=torch.cat(dsescl[i], torch.ones(dspart[subsets[i][j] ]:size(1)):fill(j-1) )
    valdses[i]=torch.cat(valdses[i],valdspart[subsets[i][j] ],1)
    valdsescl[i]=torch.cat(valdsescl[i], torch.ones(valdspart[subsets[i][j] ]:size(1)):fill(j-1) )
  end
  dsescl[i]=dsescl[i]:type("torch.CudaLongTensor")
  valdsescl[i]=valdsescl[i]:type("torch.CudaLongTensor")
  dseslabels[i]=transfrom_labels2(dsescl[i],#subsets[i],ovc[i]):cuda()
    


  nets[i]=create_net(32*32,torch.sum(ovc[i])):cuda()  
  nets[i]=cudnn.convert(nets[i], cudnn)
  optimalnets[i]=nets[i]:clone()
  optimalnets2[i]=nets[i]:clone()
  optimalaccs[i]=0;
  optimalaccs2[i]=0;
end




------------------------------------------------------------training the models
lr=0.01
timer = torch.Timer() 
for i=1,100 do

  print('epoch ' .. i .. ' time elapsed: ' .. timer:time().real)

  acc=0
  for j=1,#subsets do 
    nets[j]:training()
    shuffleIdx = torch.randperm(dses[j]:size(1)):long():type("torch.CudaLongTensor")
    dses[j] = dses[j]:index(1,shuffleIdx)
    dseslabels[j] = dseslabels[j]:index(1,shuffleIdx)
    dsescl[j] = dsescl[j]:index(1,shuffleIdx)
    collectgarbage();
  end
  for k=1,#subsets do
    minibt= torch.floor(dses[k]:size(1)/200)

    for j=1,dses[k]:size(1)/minibt do
      train(nets[k],dses[k][{{(j-1)*minibt+1,j*minibt},{},{},{}}],dseslabels[k][{{(j-1)*minibt+1,j*minibt}}],lr)
    end
  end
  


preds={}
preds2={}
for k=1,#subsets do
  minibt= torch.floor(valdses[k]:size(1)/20)
  nets[k]:evaluate()

  for j=1,valdses[k]:size(1)/minibt do
    local tempp=nn.SoftMax():cuda():forward(nets[k]:forward(valdses[k][{{(j-1)*minibt+1,j*minibt},{},{},{}}]))
    if preds[k]==nil then
      preds[k]=transfrom_predict2(tempp, #subsets[k] ,ovc[k])
      preds2[k]=transfrom_predict2sum(tempp, #subsets[k] ,ovc[k])
    else
      preds[k]=torch.cat(preds[k],transfrom_predict2(tempp, #subsets[k] ,ovc[k]) ,1)
      preds2[k]=torch.cat(preds2[k],transfrom_predict2sum(tempp, #subsets[k] ,ovc[k]) ,1)
    end
  end
  if (valdses[k]:size(1)%minibt)~=0 then
    j=torch.floor(valdses[k]:size(1)/minibt)
    local tempp=nn.SoftMax():cuda():forward(nets[k]:forward(valdses[k][{{(j)*minibt+1,valdses[k]:size(1)},{},{},{}}]))
    preds[k]=torch.cat(preds[k],transfrom_predict2(tempp, #subsets[k] ,ovc[k]) ,1)
    preds2[k]=torch.cat(preds2[k],transfrom_predict2sum(tempp, #subsets[k] ,ovc[k]) ,1)
  end
  mx,mxind=preds[k]:max(2)
  mx2,mxind2=preds2[k]:max(2)
  actpredict=mxind-1
  actpredict2=mxind2-1
  acc = torch.eq(actpredict,valdsescl[k]):sum()
  acc2 = torch.eq(actpredict2,valdsescl[k]):sum()
--  print('train acc. product: ' .. acc/valdsescl[k]:size(1))
--  print('train acc. sum: ' .. acc2/valdsescl[k]:size(1))
  if acc>=optimalaccs[k] then
    optimalnets[k]=nil
    optimalnets[k]=nets[k]:clone():clearState() 
    optimalaccs[k]=acc;
  end
  if acc2>=optimalaccs2[k] then
    optimalnets2[k]=nil
    optimalnets2[k]=nets[k]:clone():clearState()
    optimalaccs2[k]=acc2;
  end
  collectgarbage();
end
  
end

for i=1,#subsets do
    dses[i]=nil
    dsescl[i]=nil
    dseslabels[i]=nil
    valdses[i]=nil
    valdsescl[i]=nil
    nets[i]=nil
    collectgarbage();
end
----------------------------------------------------print train and test accuracies

preds={}
  preds2={}
  preds3={}
  for k=1,#subsets do
    optimalnets[k]:evaluate()
    optimalnets2[k]:evaluate()
    minibt=ds[1]:size(1)/500
    for j=1,ds[1]:size(1)/minibt do
      local tempp=nn.SoftMax():cuda():forward(optimalnets[k]:forward(ds[1][{{(j-1)*minibt+1,j*minibt},{},{},{}}]))
      local tempp2=nn.SoftMax():cuda():forward(optimalnets2[k]:forward(ds[1][{{(j-1)*minibt+1,j*minibt},{},{},{}}]))
      if preds[k]==nil then  
        preds[k]=transfrom_predict2(tempp, #subsets[k] ,ovc[k])
        preds3[k]=transfrom_predict2sum(tempp, #subsets[k] ,ovc[k])
        preds2[k]=transfrom_predict2sum(tempp2, #subsets[k] ,ovc[k])
      else
        preds[k]=torch.cat(preds[k],transfrom_predict2(tempp, #subsets[k] ,ovc[k]) ,1)
        preds3[k]=torch.cat(preds3[k],transfrom_predict2sum(tempp, #subsets[k] ,ovc[k]) ,1)
        preds2[k]=torch.cat(preds2[k],transfrom_predict2sum(tempp2, #subsets[k] ,ovc[k]) ,1)
      end
    end
  end

  sortedpreds={}
  sortedpreds2={}
  sortedpreds3={}
  tmp=1
  for i=1,#subsets do
    for j=1,#subsets[i] do
      sortedpreds[subsets[i][j]]=preds[tmp]
      sortedpreds3[subsets[i][j]]=preds3[tmp]
      sortedpreds2[subsets[i][j]]=preds2[tmp]
      tmp=tmp+1
    end
  end
  pred=sortedpreds[1]
  pred3=sortedpreds3[1]
  pred2=sortedpreds2[1]
  for j=2,#subsets do
    pred=torch.cat(pred,sortedpreds[j],2)
    pred3=torch.cat(pred3,sortedpreds3[j],2)
    pred2=torch.cat(pred2,sortedpreds2[j],2)
  end

  
  mx3,mxind3=pred3:max(2)
  localpred=pred:clone()
  for i=1,pred3:size(1) do
    if mx3[i][1]<0.99 then
      localpred[i]=pred3[i]
    else
      localpred[i]=pred[i]
    end
  end
  mx,mxind=pred:max(2)
  mx3,mxind3=localpred:max(2)
  mx2,mxind2=pred2:max(2)
  actpredict=mxind-1
  actpredict2=mxind2-1
  actpredict3=mxind3-1
  acc = torch.eq(actpredict,ds[2]):sum()
  acc2 = torch.eq(actpredict2,ds[2]):sum()
  acc3 = torch.eq(actpredict3,ds[2]):sum()
  print('product: ' .. acc/ds[2]:size(1))
  print('sum: ' .. acc2/ds[2]:size(1))
  print('local: ' .. acc3/ds[2]:size(1))





  preds={}
  preds2={}
  preds3={}
  for k=1,#subsets do
    optimalnets[k]:evaluate()
    optimalnets2[k]:evaluate()
    minibt=ds[3]:size(1)/50
    for j=1,ds[3]:size(1)/minibt do
      local tempp=nn.SoftMax():cuda():forward(optimalnets[k]:forward(ds[3][{{(j-1)*minibt+1,j*minibt},{},{},{}}]))
      local tempp2=nn.SoftMax():cuda():forward(optimalnets2[k]:forward(ds[3][{{(j-1)*minibt+1,j*minibt},{},{},{}}]))
      if preds[k]==nil then  
        preds[k]=transfrom_predict2(tempp, #subsets[k] ,ovc[k])
        preds3[k]=transfrom_predict2sum(tempp, #subsets[k] ,ovc[k])
        preds2[k]=transfrom_predict2sum(tempp2, #subsets[k] ,ovc[k])
      else
        preds[k]=torch.cat(preds[k],transfrom_predict2(tempp, #subsets[k] ,ovc[k]) ,1)
        preds3[k]=torch.cat(preds3[k],transfrom_predict2sum(tempp, #subsets[k] ,ovc[k]) ,1)
        preds2[k]=torch.cat(preds2[k],transfrom_predict2sum(tempp2, #subsets[k] ,ovc[k]) ,1)
      end
    end
  end

  sortedpreds={}
  sortedpreds2={}
  sortedpreds3={}
  tmp=1
  for i=1,#subsets do
    for j=1,#subsets[i] do
      sortedpreds[subsets[i][j]]=preds[tmp]
      sortedpreds3[subsets[i][j]]=preds3[tmp]
      sortedpreds2[subsets[i][j]]=preds2[tmp]
      tmp=tmp+1
    end
  end
  pred=sortedpreds[1]
  pred3=sortedpreds3[1]
  pred2=sortedpreds2[1]
  for j=2,#subsets do
    pred=torch.cat(pred,sortedpreds[j],2)
    pred3=torch.cat(pred3,sortedpreds3[j],2)
    pred2=torch.cat(pred2,sortedpreds2[j],2)
  end

  
  mx3,mxind3=pred3:max(2)
  localpred=pred:clone()
  for i=1,pred3:size(1) do
    if mx3[i][1]<0.99 then
      localpred[i]=pred3[i]
    else
      localpred[i]=pred[i]
    end
  end
  mx,mxind=pred:max(2)
  mx3,mxind3=localpred:max(2)
  mx2,mxind2=pred2:max(2)
  actpredict=mxind-1
  actpredict2=mxind2-1
  actpredict3=mxind3-1
  acc = torch.eq(actpredict,ds[4]):sum()
  acc2 = torch.eq(actpredict2,ds[4]):sum()
  acc3 = torch.eq(actpredict3,ds[4]):sum()
  print('product: ' .. acc/ds[4]:size(1))
  print('sum: ' .. acc2/ds[4]:size(1))




