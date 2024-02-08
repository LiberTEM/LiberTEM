import os
import stat

from libertem.common.messageconverter import MessageConverter  # NOQA: F401
from libertem.common.progress import ProgressState


class Message:
    """
    possible messages - the translation of our python datatypes to json types
    """

    def initial_state(self, jobs, datasets, analyses, compound_analyses):
        return {
            "status": "ok",
            "messageType": "INITIAL_STATE",
            "datasets": datasets,
            "jobs": jobs,
            "analyses": analyses,
            "compoundAnalyses": compound_analyses,
        }

    def cluster_conn_error(self, msg):
        return {
            "status": "error",
            "messageType": "CLUSTER_CONN_ERROR",
            "msg": msg,
        }

    def snooze(self, msg):
        return {
            "status": "ok",
            "messageType": "SNOOZE",
            "msg": msg,
        }

    def unsnooze(self, msg):
        return {
            "status": "ok",
            "messageType": "UNSNOOZE",
            "msg": msg,
        }

    def unsnooze_done(self, msg):
        return {
            "status": "ok",
            "messageType": "UNSNOOZE_DONE",
            "msg": msg,
        }

    def config(self, config):
        return {
            "status": "ok",
            "messageType": "CONFIG",
            "config": config,
        }

    def cluster_details(self, details):
        return {
            "status": "ok",
            "messageType": "CLUSTER_DETAILS",
            "details": details,
        }

    def create_dataset(self, dataset, details):
        return {
            "status": "ok",
            "messageType": "CREATE_DATASET",
            "dataset": dataset,
            "details": details,
        }

    def create_dataset_error(self, dataset, msg):
        return {
            "status": "error",
            "messageType": "CREATE_DATASET_ERROR",
            "dataset": dataset,
            "msg": msg,
        }

    def delete_dataset(self, dataset):
        return {
            "status": "ok",
            "messageType": "DELETE_DATASET",
            "dataset": dataset,
        }

    def dataset_detect(self, params, info=None):
        return {
            "status": "ok",
            "messageType": "DATASET_DETECTED",
            "datasetParams": params,
            "datasetInfo": info,
        }

    def dataset_detect_failed(self, path,
                              reason="could not automatically determine dataset format"):
        return {
            "status": "error",
            "messageType": "DATASET_DETECTION_FAILED",
            "path": path,
            "msg": reason,
        }

    def start_job(self, serialized_job, analysis_id):
        return {
            "status": "ok",
            "messageType": "JOB_STARTED",
            "job": serialized_job["id"],
            "analysis": analysis_id,
            "details": serialized_job,
        }

    def job_error(self, job_id, msg):
        return {
            "status": "error",
            "messageType": "JOB_ERROR",
            "job": job_id,
            "msg": msg,
        }

    def job_progress(self, job_id: str, state: ProgressState, event: str):
        return {
            "status": "ok",
            "messageType": "JOB_PROGRESS",
            "job": job_id,
            "details": {
                "event": event,
                "numFrames": state.num_frames_total,
                "numFramesComplete": state.num_frames_complete,
            }
        }

    def finish_job(self, serialized_job, num_images, image_descriptions):
        return {
            "status": "ok",
            "messageType": "FINISH_JOB",
            "job": serialized_job["id"],
            "details": serialized_job,
            "followup": {
                "numMessages": num_images,
                "descriptions": image_descriptions,
            },
        }

    def cancel_job(self, job_id):
        return {
            "status": "ok",
            "messageType": "CANCEL_JOB",
            "job": job_id,
        }

    def cancel_done(self, job_id):
        return {
            "status": "ok",
            "messageType": "CANCEL_JOB_DONE",
            "job": job_id,
        }

    def cancel_failed(self, job_id):
        return {
            "status": "error",
            "messageType": "CANCEL_JOB_FAILED",
            "job": job_id,
        }

    def task_result(self, job_id, num_images, image_descriptions):
        return {
            "status": "ok",
            "messageType": "TASK_RESULT",
            "job": job_id,
            "followup": {
                "numMessages": num_images,
                "descriptions": image_descriptions,
            },
        }

    def directory_listing(self, path, files, dirs, drives, places):
        def _details(item):
            return {
                "name":  item["name"],
                "size":  item["stat"].st_size,
                "ctime": item["stat"].st_ctime,
                "mtime": item["stat"].st_mtime,
                "owner": item["owner"],
            }

        return {
            "status": "ok",
            "messageType": "DIRECTORY_LISTING",
            "drives": drives,
            "places": places,
            "path": path,
            "files": [
                _details(f)
                for f in files
            ],
            "dirs": [
                _details(d)
                for d in dirs
            ],
        }

    def browse_stat_result(
        self,
        path: str,
        dirname: str,
        basename: str,
        stat_result: os.stat_result,
    ):
        return {
            "status": "ok",
            "messageType": "STAT_RESULT",
            "path": path,
            "dirname": dirname,
            "basename": basename,
            "stat": {
                "size":  stat_result.st_size,
                "ctime": stat_result.st_ctime,
                "mtime": stat_result.st_mtime,
                "isdir": bool(stat.S_ISDIR(stat_result.st_mode)),
                "isreg": bool(stat.S_ISREG(stat_result.st_mode)),
            }
        }

    def stat_failed(self, path, code, msg, alternative=None):
        return {
            "status": "error",
            "messageType": "STAT_FAILED",
            "path": path,
            "code": code,
            "msg": msg,
            "alternative": alternative,
        }

    def browse_failed(self, path, code, msg, alternative=None):
        return {
            "status": "error",
            "messageType": "DIRECTORY_LISTING_FAILED",
            "path": path,
            "code": code,
            "msg": msg,
            "alternative": alternative,
        }

    def create_analysis(self, uuid, dataset_uuid, analysis_type, parameters):
        return {
            "status": "ok",
            "messageType": "ANALYSIS_CREATED",
            "analysis": uuid,
            "dataset": dataset_uuid,
            "details": {
                "analysisType": analysis_type,
                "parameters": parameters,
            }
        }

    def update_analysis(self, uuid, dataset_uuid, analysis_type, parameters):
        return {
            "status": "ok",
            "messageType": "ANALYSIS_UPDATED",
            "analysis": uuid,
            "dataset": dataset_uuid,
            "details": {
                "analysisType": analysis_type,
                "parameters": parameters,
            }
        }

    def analysis_removed(self, uuid):
        return {
            "status": "ok",
            "messageType": "ANALYSIS_REMOVED",
            "analysis": uuid,
        }

    def analysis_removal_failed(self, uuid, msg):
        return {
            "status": "error",
            "messageType": "ANALYSIS_REMOVAL_FAILED",
            "analysis": uuid,
            "msg": msg,
        }

    def compound_analysis_created(self, serialized):
        msg = {
            "status": "ok",
            "messageType": "COMPOUND_ANALYSIS_CREATED",
        }
        msg.update(serialized)
        return msg

    def compound_analysis_updated(self, serialized):
        msg = {
            "status": "ok",
            "messageType": "COMPOUND_ANALYSIS_UPDATED",
        }
        msg.update(serialized)
        return msg

    def compound_analysis_removed(self, uuid):
        msg = {
            "status": "ok",
            "messageType": "COMPOUND_ANALYSIS_REMOVED",
            "compoundAnalysis": uuid,
        }
        return msg
