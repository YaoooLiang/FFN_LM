#
import logging
import os
try:
    import Queue as queue
except ImportError:  # for Python 3 compat
    import queue
import threading
import time
import torch
from torch.autograd import Variable
import numpy as np


# pylint:disable=g-import-not-at-top
try:
    import thread
except ImportError:  # for Python 3 compat
    import _thread as thread
# pylint:enable=g-import-not-at-top


class BatchExecutor(object):
    """Base class for FFN executors."""

    def __init__(self, model, batch_size):
        self.model = model

        self.batch_size = batch_size
        self.active_clients = 0

        # Cache input/output sizes.
        self._input_seed_size = np.array(model.input_size[::-1]).tolist()
        self._input_image_size = np.array(model.input_size[::-1]).tolist()

    def start_server(self):
        raise NotImplementedError()

    def stop_server(self):
        raise NotImplementedError()

    def start_client(self):
        """Registers a new client.

        Returns:
          client ID
        """
        raise NotImplementedError()

    def finish_client(self, client_id):
        """Deregisters a client."""
        raise NotImplementedError()

    def predict(self, client_id, seed, image):
        raise NotImplementedError()

    def _run_executor(self):
        raise NotImplementedError()

    def _run_executor_log_exceptions(self):
        """Runs the main loop of the executor.

        Logs any exceptions and re-raises them.
        """
        try:
            self._run_executor()
        except Exception as e:  # pylint:disable=broad-except
            logging.exception(e)
            # If the executor fails, the whole process becomes useless and we need
            # to make sure it gets terminated.
            thread.interrupt_main()
            time.sleep(10)
            os._exit(1)  # pylint:disable=protected-access

    @property
    def num_devices(self):
        return 1


class ThreadingBatchExecutor(BatchExecutor):
    """Thread-based BatchExecutor.

    The intended use is to have multiple threads sharing the same executor
    object with:
      - a server thread started with `start_server`
      - each client running in its own thread.

    It is recommended to start the client threads as daemons, so that failures
    of the server thread will result in termination of the whole program.

    Note that the number of clients can (and for efficient utilization of ML
    accelerators, should) exceed the batch size. This makes sense to do even
    if the batch size is 1.
    """

    def __init__(self, model, batch_size, expected_clients=1):
        super(ThreadingBatchExecutor, self).__init__(model, batch_size)
        self._lock = threading.Lock()
        self.outputs = {}  # Will be populated by Queues as clients register.
        # Used by clients to communiate with the executor. The protocol is
        # as follows:
        #  - 'exit': indicates a request to kill the executor
        #  - N >= 0: registration of a new client with the specified ID
        #  - N < 0: deregistration of an existing client with ID -N - 1
        #  (client_id, seed, image, fetches): request to perform inference
        self.input_queue = queue.Queue()

        # Total clients seen during the lifetime of the executor.
        self.total_clients = 0

        # This many clients need to register themselves during the lifetime of
        # the executor in order for it be allowed to terminate.
        self.expected_clients = expected_clients

        # Arrays fed to TF.
        self.input_seed = np.zeros([batch_size] + [1] + self._input_seed_size,
                                   dtype=np.float32)
        self.input_image = np.zeros([batch_size] + [3] + self._input_image_size,
                                    dtype=np.float32)
        self.th_executor = None

    def start_server(self):
        """Starts the server which will evaluate TF models.

        The server will automatically terminate after no more clients are
        registered, and after at least one client has registered and
        deregistered.
        """
        if self.th_executor is None:
            self.th_executor = threading.Thread(
                target=self._run_executor_log_exceptions)
            self.th_executor.start()

    def stop_server(self):
        logging.info('Requesting executor shutdown.')
        self.input_queue.put('exit')
        self.th_executor.join()
        logging.info('Executor shutdown complete.')

    def _run_executor(self):
        """Main loop of the server thread which runs TF code."""
        self._curr_infeed = 0
        logging.info('Executor starting.')

        while self.active_clients or self.total_clients < self.expected_clients:

            ready = []
            while (len(ready) < min(self.active_clients, self.batch_size) or
                   not self.active_clients):
                try:
                    data = self.input_queue.get(timeout=5)
                except queue.Empty:
                    continue
                if data == 'exit':
                    logging.info('Executor shut down requested.')
                    return
                elif isinstance(data, int):
                    client_id = data
                    if client_id >= 0:
                        self.total_clients += 1
                        self.active_clients += 1
                        logging.info('client %d starting', client_id)
                    else:
                        logging.info('client %d terminating', -client_id - 1)
                        self.active_clients -= 1
                else:
                    client_id, seed, image = data
                    l = len(ready)
                    self.input_seed[l, ...] = seed
                    self.input_image[l, ...] = image
                    ready.append(client_id)

            if ready:
                self._schedule_batch(ready)

        logging.info('Executor terminating.')

    def _schedule_batch(self, client_ids):
        """Schedules a single batch for execution."""
        try:
            input_image = torch.from_numpy(self.input_image).float()
            input_seed = torch.from_numpy(self.input_seed).float()
            input_data = torch.cat([input_image, input_seed], dim=1)
            input_data = Variable(input_data.cuda())

            logits = self.model(input_data)
            ret = (input_seed.cuda() + logits).detach().cpu().numpy()
        except Exception as e:  # pylint:disable=broad-except
            logging.exception(e)
            # If calling TF didn't work (faulty hardware, misconfiguration, etc),
            # we want to terminate the whole program.
            thread.interrupt_main()
            raise e

        with self._lock:
            for i, client_id in enumerate(client_ids):
                try:
                    self.outputs[client_id].put(ret[i, ...])
                except KeyError:
                    # This could happen if a client unregistered itself
                    # while inference was running.
                    pass

    def start_client(self):
        with self._lock:
            if not self.outputs:
                client_id = 0
            else:
                client_id = max(self.outputs.keys()) + 1

            self.outputs[client_id] = queue.Queue()

        self.input_queue.put(client_id)
        return client_id

    def finish_client(self, client_id):
        self.input_queue.put(-1 - client_id)
        with self._lock:
            del self.outputs[client_id]

    def predict(self, client_id, seed, image):
        self.input_queue.put((client_id, seed, image))

        ret = self.outputs[client_id].get()

        return ret
