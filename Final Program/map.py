# from jetracer.nvidia_racecar import NvidiaRacecar
# car = NvidiaRacecar()
# car.steering_offset=0
# car.steering = 0
# car.throttle_gain = 0.3
#min Steering -1 Right max 1 Left
#Forward -0.5 backward 0.5
def getAction(num):
    For = -0.5
    Back = 0.5
    Left = 1
    Right = -1
    StepSteering = 1/3
    steering = None
    throttle = None
    if num == 0:
        steering = Left
        throttle = Back
    elif num == 1:
        steering = Left
        throttle = 0
    elif num == 2:
        steering = Left
        throttle = For
    #-20
    elif num == 3:
        steering = Left-StepSteering
        throttle = Back
    elif num == 4:
        steering = Left-StepSteering
        throttle = 0
    elif num == 5:
        steering = Left-StepSteering
        throttle = For
    #-10
    elif num == 6:
        steering = Left-(2*StepSteering)
        throttle = Back
    elif num == 7:
        steering = Left-(2*StepSteering)
        throttle = 0
    elif num == 8:
        steering = Left-(2*StepSteering)
        throttle = For
    #0
    elif num == 9:
        steering = 0
        throttle = Back
    elif num == 10:
        steering = 0
        throttle = 0
    elif num == 11:
        steering = 0
        throttle = For
    #10
    elif num == 12:
        steering = Right+(2*StepSteering)
        throttle = Back
    elif num == 13:
        steering = Right+(2*StepSteering)
        throttle = 0
    elif num == 14:
        steering = Right+(2*StepSteering)
        throttle = For
    #20
    elif num == 15:
        steering = Right+(StepSteering)
        throttle = Back
    elif num == 16:
        steering = Right+(StepSteering)
        throttle = 0
    elif num == 17:
        steering = Right+(StepSteering)
        throttle = For
    #10
    elif num == 18:
        steering = Right
        throttle = Back
    elif num == 19:
        steering = Right
        throttle = 0
    elif num == 20:
        steering = Right
        throttle = For
    return [throttle,steering]
    