from typing import List

class Result():
    """
    This is a version of the monad Result

    Habitually, we should not be able to access the ok value, or the error value.
    It's one of the core concept of a monad. 

    A monad is like a box. You always need to pass the box to other function, and
    when you want to open the box to see what's inside, you absolutely need to 
    tell what to do if it's an error and what to do if it's a success(ok)
    
    For now, we can access the ok and error value for ease of use, but we should use
    the match function that make mandatory to tell the result what to do in 
    both cases.

    By being able to use the ok and error value, it makes the Result class to be
    used with less restreint.

    """
    __create_key = object()

    def __init__(self, create_key, ok=None, error=None):
        assert(create_key == Result.__create_key), \
            "Result objects must be created using Result.create"

        self.ok = ok
        self.error = error

    @classmethod
    def ok(cls, ok):
        return Result(cls.__create_key, ok=ok)

    @classmethod
    def error(cls, error):
        return Result(cls.__create_key, error=error)

    def is_ok(self):
        return self.ok is not None

    def is_error(self):
        return self.error is not None

    def match(self, ok_function, error_function):
        """
        The method match demands you to tell what to do in both cases,
        without giving you especially both values. 
        
        Concept of the monad(box).

        """
        if self.is_ok():
            ok_function(self.ok)
        else:
            error_function(self.error)

class ResultUtils():

    @staticmethod
    def map_ok(result:Result, map_function):
        """
        Maps a result by passing a function of type a -> b

        Parameters:
        map_function: function of type a -> b

        Returns a result with ok value equals to the mapping function over the ok value
        Returns the result if the result is error

        Example:

        Result(ok=a:str) -> map_ok(a:str -> b:int) -> Result(ok=b:int)

        Railway

        If is_ok
        OK  = a:str  ------- a:str -> b:int -------- Ok  = b:int
        Err = None   ------------------------------- Err = None

        If is_error
        OK  = None    ---------- a:str -> b:int --------- Ok  = None
        Err = msg:str ----------------------------------- Err = msg:str


        """
        if result.is_ok():
            return Result.ok (map_function(result.ok))
        else:
            # Error result propagation. If we are in error state, we stay in error state
            return result

    @staticmethod
    def map_error(result:Result, map_function):
        """
        Maps a result by passing a function of type a -> b

        Parameters:
        map_function: function of type a -> b

        Returns a result with error value equals to the mapping function over the error value
        Returns the result if the result is ok

        Result(err=ex:Exception) -> map_error(ex:Exception -> msg:str) -> Result(err=msg:str)

        Railway
        
        If is_ok
        OK  = None         ========================================= Ok  = b:int
        Err = ex:Exception ======== ex:Exception -> msg:str ======== Err = msg:str

        If is_error
        OK  = a:str ========================================= Ok  = a:str
        Err = None  ========= ex:Exception -> msg:str ======= Err = None
        """
        if result.is_error():
            return Result.error (map_function(result.error))
        else:
            # Ok result propagation. If we are in Ok state, we stay in Ok state
            return result

    @staticmethod
    def bi_map(result:Result, map_ok_function, map_error_function):
        """
        Syntax sugar 

        It's simply a way to handle map_ok, and map_error in one function call
        """
        if result.is_ok():
            return ResultUtils.map_ok(result, map_ok_function)
        else:
            return ResultUtils.map_error(result, map_error_function)

    @staticmethod
    def bind_ok(result:Result, function):
        """
        Binds a result with a function of type a -> Result(ok, error)

        Parameters:
        function: function of type a -> Result(ok, error)

        Sometimes we want to plug a function of type a -> Result(ok, error) in the railway system.
        Per example, using map would return:

        MAP_OK
        OK  = success:Response ======= map_ok(success:Response -> result:Result(ok=a:str, err=None)) ======= Ok = result:Result(ok=a:str, err=None)
        Err = None ========================================================================================= Err = None

        The returned type is Result(ok=result:Result(ok=a:str, err=None), err=None), so a result in a result. It is not what we want, we
        want a single result like this:

        BIND_OK
        OK  = success:Response ========== bind_ok(success:Response -> result:Result(ok=a:str,  ============= Ok = a:str
        Err = None **This Err is not returned anymore**                             err=None)) ============= Err = None

        """
        if result.is_ok():
            return function(result.ok)
        else:
            # Error result propagation. If we are in error state, we stay in error state
            return result

    @staticmethod
    def bind_error(result:Result, function):
        """
        Take a look at bind_ok.

        It's same but when we are in error state. 

        This fonction allow a result in error state to return in a success state.

        """
        if result.is_error():
            return function(result.error)
        else:
            return result

    @staticmethod
    def tee_error(result:Result, side_effect_function):
        """
        tee_error is used for side effect function, functions that returns None.

        The most common side effect function is the print function. 

        Example:

        TEE_ERROR
        OK=None     ================================================ Ok=None
        Err=ex:Exception =========================================== Err=ex:Exception 
                           tee_error(print(ex.Message)) -> None

        It will print the error without altering the result returned; to be use by another process.
        """
        if result.is_error():
            side_effect_function(result.error)
        return result

    @staticmethod
    def tee_ok(result:Result, side_effect_function):
        """
        tee_ok is used for side effect function, functions that returns None.

        The most common side effect function is the print function. 

        Example:

        TEE_OK
                                    tee_ok(print(f'The dataset {response.dataset_id} has been created with success.)) -> None
        OK= response:Response ================================================================================================== Ok=response:Response
        Err=None =============================================================================================================== Err=None

        We didn't apply changes to the result, we just called a side effect and returned the result; to be use by another process needing the response.
                        
        """
        if result.is_ok():
            side_effect_function(result.ok)
        return result

    @staticmethod
    def bi_tee(result:Result, side_effect_ok, side_effect_error):
        """
        Syntax sugar

        To declare to side effect for both result states in one function call.
        """
        if result.is_ok():
            side_effect_ok(result.ok)
        else:
            side_effect_error(result)
        return result


    @staticmethod
    def extract(results: List[Result]):
        values = []
        for result in results:
            if result.is_ok():
                values.append(result.ok)
            
        return values