# Error Message 返回
from flask import jsonify

class ERROR:
    def __init__(self):
        self.error_list = {
            "1001": "BAD_ARGUMENTS",
            "1002": "MISSING_ARGUMENTS",
            "1003": "INTERNAL_ERROR",
            "1004": "FIlE_ERROR",
            "1005": "UPLOAD_ERROR",
            "1006": "FACE_TOKEN_ERROR",
            "1007": "FACE_INFO_ERROR"
        }
    
    def get_error(self, error_code, error_msg):
        if error_msg is not None:
            return jsonify({
                "sucess": False, 
                "error_code": error_code, 
                "error_msg": error_msg
            })
        elif error_code in self.error_list.keys():
            return jsonify({
                "sucess": False, 
                "error_code": error_code, 
                "error_msg": self.error_list[error_code]
            })
        else:
            return jsonify({
                "sucess": False, 
                "error_code": error_code, 
                "error_msg": "undefind error code"
            })
