import json
import os
import socket
import struct

import imageio
import numpy as np
import object_detection.object_detector as obj
import action_detection.action_detector as act


MAGIC_NUMBER = bytes([66])
HOST = ''
PORT = 8089
FILENAME = 'filename'
NUM_INPUT_FRAMES = 32
ACTION_CLASSES_TO_PRINT = 5


class Server:
    def __init__(self, conn):
        self.conn = conn

    def recv_n_bytes(self, n):
        """ Convenience method for receiving exactly n bytes from
        socket (assuming it's open and connected).
        """

        # based on https://docs.python.org/3.4/howto/sockets.html
        chunks = []
        bytes_recd = 0
        while bytes_recd < n:
            chunk = self.conn.recv(n - bytes_recd)
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd += len(chunk)
        return b''.join(chunks)

    def consume_magic_number(self):
        response = self.recv_n_bytes(1)
        return response == MAGIC_NUMBER


    def run_cv(self, video_path):
        obj_detection_model =  'ssd_mobilenet_v2_coco_2018_03_29'
        obj_detection_graph = os.path.join(
            "object_detection", "weights", obj_detection_model, "frozen_inference_graph.pb")

        print("Loading object detection model at %s" % obj_detection_graph)

        obj_detector = obj.Object_Detector(obj_detection_graph)
        tracker = obj.Tracker(timesteps=NUM_INPUT_FRAMES)

        reader = imageio.get_reader(video_path, 'ffmpeg')
        W, H = reader.get_meta_data()['size']
        print(W, H)
        T = tracker.timesteps

        act_detector = act.Action_Detector('soft_attn', timesteps=NUM_INPUT_FRAMES)

        ckpt_name = 'model_ckpt_soft_attn_pooled_cosine_drop_ava-130'

        input_frames, temporal_rois, temporal_roi_batch_indices, cropped_frames = act_detector.crop_tubes_in_tf([T,H,W,3])    
        rois, roi_batch_indices, pred_probs = act_detector.define_inference_with_placeholders_noinput(cropped_frames)

        ckpt_path = os.path.join('action_detection', 'weights', ckpt_name)
        act_detector.restore_model(ckpt_path)

        for frame_number in range(NUM_INPUT_FRAMES):
            cur_img = reader.get_data(frame_number)
            expanded_img = np.expand_dims(cur_img, axis=0)

            detection_list = obj_detector.detect_objects_in_np(expanded_img)
            detection_info = [info[0] for info in detection_list]
            tracker.update_tracker(detection_info, cur_img)
            no_actors = len(tracker.active_actors)

            if tracker.active_actors and len(tracker.frame_history) >= tracker.timesteps:

                cur_input_sequence = np.expand_dims(
                    np.stack(tracker.frame_history[-tracker.timesteps:], axis=0), axis=0)

                rois_np, temporal_rois_np = tracker.generate_all_rois()
                if no_actors > 14:
                    no_actors = 14
                    rois_np = rois_np[:14]
                    temporal_rois_np = temporal_rois_np[:14]

                feed_dict = {
                    input_frames:cur_input_sequence,
                    temporal_rois: temporal_rois_np,
                    temporal_roi_batch_indices: np.zeros(no_actors),
                    rois:rois_np, 
                    roi_batch_indices:np.arange(no_actors)
                }
                run_dict = {'pred_probs': pred_probs}
                out_dict = act_detector.session.run(run_dict, feed_dict=feed_dict)
                probs = out_dict['pred_probs']

                # associate probs with actor ids
                for bb in range(no_actors):
                    act_probs = probs[bb]
                    order = np.argsort(act_probs)[::-1]
                    cur_actor_id = tracker.active_actors[bb]['actor_id']
                    print("Person %i" % cur_actor_id)
                    for pp in range(ACTION_CLASSES_TO_PRINT):
                        print('\t %s: %.3f' % (act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))

        print('End CV loop')


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(10)
    conn, _ = s.accept()
    print('Device connected')

    server = Server(conn)
    while True:
        found_magic = server.consume_magic_number()
        if not found_magic:
            raise Exception('Client did not send magic number')

        header_size = struct.unpack("!I", server.recv_n_bytes(4))[0]
        header_raw = server.recv_n_bytes(header_size)
        header = json.loads(header_raw.decode('ascii'))

        video_size = struct.unpack("!I", server.recv_n_bytes(4))[0]

        filename = os.path.basename(header[FILENAME])

        with open(filename, 'wb') as video_file:
            video_file.write(server.recv_n_bytes(video_size))
            
        print('Video received')

        server.run_cv(filename)


if __name__ == '__main__':
    main()
