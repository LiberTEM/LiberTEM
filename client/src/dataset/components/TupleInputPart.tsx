import * as React from "react";

interface TupleInputPartProps {
    name: string,
    id: string,
    tupleKey: number,
    value: number,
    tupleInputChangeHandle: (idx: number, value: string) => void,
    onBlur: (idx: number, value: string) => void,
    commaPressHandle: (idx: number) => void,
    tupleRef: (ref:HTMLInputElement) => void,
}

const TupleInputPart: React.FC<TupleInputPartProps> = ({ name, id, tupleKey, tupleInputChangeHandle, value, commaPressHandle, tupleRef, onBlur }) => {

   const onPartChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      tupleInputChangeHandle(tupleKey, e.target.value);
   }

   const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
       onBlur(tupleKey, e.target.value);
   }

   const onCommaPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
     if(e.keyCode === 188) {
       commaPressHandle(tupleKey);
     }
   }

   return <input type="number" name={name} id={id} key={tupleKey} onChange={onPartChange} value={value || ""} onKeyDown={onCommaPress} ref={tupleRef} onBlur={handleBlur} />;
}

export default TupleInputPart;
