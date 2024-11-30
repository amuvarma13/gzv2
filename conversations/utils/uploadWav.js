import { ref, uploadBytes, getDownloadURL } from "firebase/storage";
import { storage } from "../../../firebase";
import { doc, updateDoc } from "firebase/firestore";
import { db } from "../../../firebase";


export const uploadWav = async ({
    blob, 
    blockId, 
    conversationId, 
    user, 

}) => {
    const storageRef = ref(storage, `users/${user.uid}/conversations/${conversationId}/blocks/${blockId}.wav`);
    await uploadBytes(storageRef, blob);
    const downloadURL = await getDownloadURL(storageRef);
    const userDocRef = doc(db, 'users', user.uid);
    const conversationDocRef = doc(userDocRef, 'sentences', conversationId);
    await updateDoc(doc(conversationDocRef, 'transcripts', blockId), { 
        audioPath: storageRef.fullPath, 
        downloadURL,
        recorded: true
    });
    return storageRef.fullPath;
}
