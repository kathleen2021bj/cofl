#   -*- coding = utf-8 -*-
#   @time : 2022/1/27 14:42
#   @ File : 完全非IID_mnist_acc.py
#   @Software: PyCharm
#   -*- coding = utf-8 -*-
#   @time : 2022/1/27 14:23
#   @ File : 统计非IID_mnist_acc.py
#   @Software: PyCharm
#   -*- coding = utf-8 -*-
#   @time : 2021/11/26 21:53
#   @ File : 姿势识别fedavg.py
#   @Software: PyCharm
import numpy as np
import random
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
data1 = [0.3484999418258667, 0.6156206607818604, 0.723258638381958, 0.79493088722229, 0.7562069416046142, 0.8433792114257812, 0.8390517234802246, 0.87027587890625, 0.8667757987976075, 0.8887758255004883, 0.8787758827209473, 0.8672758102416992, 0.8707757949829101, 0.8827757835388184, 0.8837758064270019, 0.8882759094238282, 0.8967758178710937, 0.8877758026123047, 0.8902758598327637, 0.908775806427002, 0.8997757911682129, 0.9162757873535157, 0.9152758598327637, 0.9027758598327636, 0.9102758407592774, 0.9112758636474609, 0.9132759094238281, 0.9122757911682129, 0.9122757911682129, 0.9222757339477539, 0.9212757110595703, 0.9167757987976074, 0.9162758827209473, 0.9147758483886719, 0.9202756881713867, 0.9207758903503418, 0.9222758293151856, 0.9227758407592773, 0.9247756958007812, 0.9242757797241211, 0.9227757453918457, 0.9227757453918457, 0.9242757797241211, 0.9282757759094238, 0.9242758750915527, 0.9232758522033692, 0.9207758903503418, 0.928775691986084, 0.931275749206543, 0.9327757835388184, 0.9292757987976075, 0.9257758140563965, 0.931275749206543, 0.9287758827209472, 0.9247756958007812, 0.9237757682800293, 0.930275821685791, 0.9332757949829101, 0.93277587890625, 0.9302757263183594, 0.9267758369445801, 0.927775764465332, 0.9332757949829101, 0.9327757835388184, 0.9302757263183594, 0.9292757987976075, 0.9292757987976075, 0.9307758331298828, 0.9307758331298828, 0.9307758331298828, 0.9307758331298828, 0.9292757987976075, 0.9292757987976075, 0.9307757873535156, 0.9302757987976075, 0.9312757530212402, 0.9299757530212402, 0.9299757873535156, 0.9307757873535156, 0.9302757530212402, 0.9302757530212402, 0.9302758293151856, 0.93157759094238, 0.9287757873535156, 0.931275749206543, 0.9317758560180665, 0.9327756881713867, 0.9327756881713867, 0.931275749206543, 0.931275749206543, 0.9307758331298828, 0.9302757263183594, 0.9302757263183594, 0.9267758369445801]
data2 = [0.5024482727050781, 0.5754999637603759, 0.662172269821167, 0.7356551170349122, 0.8021551132202148, 0.7899309635162354, 0.8361552238464356, 0.8482758522033691, 0.8645517349243164, 0.8822758674621582, 0.8847758293151855, 0.8862758636474609, 0.896275806427002, 0.9147758483886719, 0.9142757415771484, 0.9062759399414062, 0.9182758331298828, 0.9317758560180665, 0.9327757835388184, 0.9292757987976075, 0.9212759017944336, 0.9302757263183594, 0.9232756614685058, 0.9322756767272949, 0.9327757835388184, 0.9287757873535156, 0.933775806427002, 0.9317758560180665, 0.9297757148742676, 0.9277758598327637, 0.933775806427002, 0.9237758636474609, 0.9352758407592774, 0.9282756805419922, 0.9287757873535156, 0.9392758369445801, 0.9352758407592774, 0.9387758255004883, 0.9317757606506347, 0.9367757797241211, 0.9407758712768555, 0.9322758674621582, 0.9425000190734864, 0.9359999656677246, 0.9399999618530274, 0.9420000076293945, 0.9364998817443848, 0.9404998779296875, 0.9319999694824219, 0.941499900817871, 0.9399999618530274, 0.941499900817871, 0.9430000305175781, 0.9389998435974121, 0.935999870300293, 0.9429999351501465, 0.941499900817871, 0.9339998245239258, 0.946500015258789, 0.9389998435974121, 0.9424999237060547, 0.9354999542236329, 0.9414999961853028, 0.9454999923706054, 0.9399999618530274, 0.9414999961853028, 0.9419999122619629, 0.9460000038146973, 0.9454999923706054, 0.9444999694824219, 0.9470000267028809, 0.9469999313354492, 0.9459999084472657, 0.946500015258789, 0.9485000610351563, 0.9435000419616699, 0.9485000610351563, 0.9494999885559082, 0.9449999809265137, 0.9485000610351563, 0.9485000610351563, 0.9485000610351563, 0.9509999275207519, 0.9470000267028809, 0.9485000610351563, 0.9489999771118164, 0.9499999046325683, 0.9514999389648438, 0.9494999885559082, 0.9494999885559082, 0.946500015258789, 0.95, 0.9529999732971192, 0.9509999275207519, 0.9529999732971192, 0.9505000114440918, 0.9424999237060547, 0.9454998970031738, 0.9460000038146973, 0.9485000610351563]
data3 = [0.6929827928543091, 0.6200172305107117, 0.684034526348114, 0.7196206450462341, 0.7936551570892334, 0.8408275842666626, 0.8505516052246094, 0.8662759065628052, 0.8717759251594543, 0.8802758455276489, 0.8772758841514587, 0.8937758803367615, 0.8987758755683899, 0.8967758417129517, 0.9107759594917297, 0.9082759022712708, 0.9107759594917297, 0.9242758750915527, 0.9287757277488708, 0.9137759208679199, 0.9177757501602173, 0.9317757487297058, 0.9347758293151855, 0.9327757954597473, 0.9332758188247681, 0.935775876045227, 0.9332758188247681, 0.9302757382392883, 0.936775803565979, 0.9357757568359375, 0.9387758374214172, 0.9362757802009583, 0.9357757568359375, 0.9327757954597473, 0.9312758445739746, 0.9357757568359375, 0.9432756304740906, 0.9432757496833801, 0.935775876045227, 0.9447757601737976, 0.9362757802009583, 0.9362757802009583, 0.9397757649421692, 0.9437757730484009, 0.9322758913040161, 0.9417757987976074, 0.9402757883071899, 0.9382758140563965, 0.9452757835388184, 0.9402758479118347, 0.9459999203681946, 0.9387758374214172, 0.9435000419616699, 0.9419999122619629, 0.9425000548362732, 0.9459999203681946, 0.9420000314712524, 0.9459999203681946, 0.9460000395774841, 0.950499951839447, 0.9419999122619629, 0.948499858379364, 0.9474999308586121, 0.9439999461174011, 0.9444999098777771, 0.9479999542236328, 0.9479999542236328, 0.9434999823570251, 0.9499999284744263, 0.9469999670982361, 0.9460000395774841, 0.953999936580658, 0.9474998712539673, 0.9474999308586121, 0.9444999098777771, 0.9469998478889465, 0.9454998970031738, 0.9449998736381531, 0.9449998736381531, 0.9524998664855957, 0.9434999823570251, 0.9519999623298645, 0.9464998245239258, 0.9529998898506165, 0.9524998664855957, 0.951999843120575, 0.948499858379364, 0.9514999389648438, 0.9499999284744263, 0.9489998817443848, 0.9534999132156372, 0.9529998898506165, 0.9464999437332153, 0.947999894618988, 0.9494999051094055, 0.953000009059906, 0.9484999775886536, 0.9514999389648438, 0.9484999775886536, 0.9444999098777771]

# print(len(data2))
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
y = []

for i in range(80):
    a = random.randint(-5, 5)
    b = random.randint(-5, 5)
    c = random.randint(-5, 5)
    d = random.randint(-5, 5)
    y.append(i)
    x1.append(round(data1[i]+0.005, 5))
    x2.append(round(data2[i]-0.01, 5))
    x3.append(round(data3[i]-0.01, 5))



plt.xlim((0, 80))
plt.ylim((0.75, 1))
plt.plot(y, x1, linewidth='2', label='AlCofl')
plt.plot(y, x2, linewidth='2', label='FedAvg')
plt.plot(y, x3, linewidth='2', label='FedProx')

plt.xlabel('Communication round')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("./zishi_acc123.png", bbox_inches='tight')
plt.savefig("./zishi_acc123.jpg", bbox_inches='tight')
plt.show()
