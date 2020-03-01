from apscheduler.schedulers.blocking import BlockingScheduler
import predict
import threading

sched = BlockingScheduler()


@sched.scheduled_job('cron', day='*')
def scheduled_job():
    thread1 = threading.Thread(target=predict.refresh, args=('confirmed',))
    thread1.start()
    # thread2 = threading.Thread(target=predict.refresh, args=('deaths',))
    # thread2.start()
    # thread3 = threading.Thread(target=predict.refresh, args=('recovered',))
    # thread3.start()
    thread4 = threading.Thread(target=predict.refresh, args=('curr_confirmed',))
    thread4.start()
    # thread5 = threading.Thread(target=predict.refresh, args=('curr_deaths',))
    # thread5.start()
    # thread6 = threading.Thread(target=predict.refresh, args=('curr_recovered',))
    # thread6.start()
    return "Successfully started refreshing data and predictions."


sched.start()
