import os
import cv2
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
from PIL import Image
import logging

from final_score_calculator import get_possible_question_paper_ids
from find_marked_omr import find_score_for_imr




TOKEN = "8195046189:AAGAj9EHKJBJZF9HvGEooFDb5e4Engswx9g"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Dictionary to store user-specific question paper IDs
user_qpid = {}



# Command to set Question Paper ID
# async def set_qpid(update: Update, context: CallbackContext):
#     user_id = update.message.chat_id
#
#     if not context.args:
#         await update.message.reply_text("Usage: /set_question_paper_id <question_paper_id>")
#         logging.warning(f"User {user_id} tried to set QPID without an argument.")
#         return
#
#     qpid = context.args[0]
#     user_qpid[user_id] = qpid
#     logging.info(f"User {user_id} set Question Paper ID to {qpid}.")
#     await update.message.reply_text(f"✅ Question Paper ID set to: {qpid}\nNow send an OMR image.")


def format_score_response(total_score, ans_matching):
    message = f"📊  Test Results 📊\n"
    message += f"✅  Score: {total_score}\n\n"

    wrong_count = 0
    blank_count = 0
    correct_count = 0

    for key, ans in ans_matching.items():
        type_of_ans, _ , _ = ans
        if type_of_ans == "WRONG":
            wrong_count += 1
        if type_of_ans == "BLANK":
            blank_count += 1
        if type_of_ans == "CORRECT":
            correct_count += 1

    # wrong_count = sum(1 for ans in ans_matching if ans_matching[ans][0] == "WRONG")
    # blank_count = sum(1 for ans in ans_matching if ans_matching[ans][0] == "BLANK")
    # Show detailed results including correct, incorrect, and blank answers
    ans_details = "📊 Detailed Results\n"
    ans_details += f"{'Q No':<5} | {'Status':<7} | {'Marked':<6} | {'Expected':<8}\n"
    ans_details += "-" * 36 + "\n"

    for q_no, (status, marked, actual) in ans_matching.items():
        if status == "CORRECT":
            ans_details += f"Q{q_no:<5} | {'✅':<7} | {marked.upper():<6} | {actual.upper():<8}\n"
        elif status == "WRONG":
            ans_details += f"Q{q_no:<5} | {'❌':<7} | {marked.upper():<6} | {actual.upper():<8}\n"
        elif status == "BLANK":
            ans_details += f"Q{q_no:<5} | {'⚪':<7} | {'':<6} | {actual.upper():<8}\n"


    message += f"\n📌 Summary: \n✅ {correct_count} correct \n❌ {wrong_count} wrong \n⚪ {blank_count} not attempted"
    message += "\n💪 Keep practicing!"

    return message, ans_details



async def set_qpid(update: Update, context: CallbackContext):
    user_id = update.message.chat_id
    QUESTION_PAPER_IDS = get_possible_question_paper_ids()
    keyboard = [
        [InlineKeyboardButton(qpid, callback_data=f"set_qpid_{qpid}")]
        for qpid in QUESTION_PAPER_IDS
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "📄 Please select a Question Paper ID from the list below:",
        reply_markup=reply_markup
    )

async def button_click(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()

    user_id = query.message.chat_id
    selected_qpid = query.data.replace("set_qpid_", "")
    user_qpid[user_id] = selected_qpid

    logging.info(f"User {user_id} set Question Paper ID to {selected_qpid}.")
    await query.edit_message_text(f"✅ Question Paper ID set to: {selected_qpid}\nNow send an OMR image.")


# Handler for OMR images
async def handle_image(update: Update, context: CallbackContext):

    user_id = update.message.chat_id

    if user_id not in user_qpid:
        await update.message.reply_text("⚠️ Please set the Question Paper ID first using /set_question_paper_id <id>")
        logging.warning(f"User {user_id} tried to upload an OMR image without setting QPID.")
        return
    qpid = user_qpid[user_id]
    file = await update.message.photo[-1].get_file()
    file_path = f"omr_{user_id}.jpg"
    await file.download_to_drive(file_path)

    abs_file_path = os.path.abspath(file_path)

    logging.info(f"User {user_id} uploaded an OMR image. Saved as {file_path}. Processing...")

    try:
        await update.message.reply_text(f"✅ Processing your OMR sheet for question paper {qpid} please wait...")
        score, answer_matching = find_score_for_imr(abs_file_path, template_for_questions, template_for_omr, question_paper_id=qpid)
        summary, ans_details = format_score_response(score, answer_matching)
        logging.info(f"User {user_id} got a score of {score} for QPID {qpid}.")
        await update.message.reply_text(summary)
        await update.message.reply_text(ans_details)
        await update.message.reply_photo(photo="./debug/05_marked_answers.jpg")
    except Exception as e:
        logging.error(f"Error processing OMR image for user {user_id}: {e}")
        await update.message.reply_text("❌ Error processing OMR image. Please try again.")


# Main function to run the bot
def main():



    logging.info("Bot is starting...")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("set_question_paper_id", set_qpid))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(CallbackQueryHandler(button_click))  # Handle button clicks
    logging.info("Bot is running and waiting for commands...")
    app.run_polling()


if __name__ == "__main__":
    template_for_questions = cv2.imread("./templates/template_for_questions.jpeg")
    template_for_omr = cv2.imread("./templates/template_for_whole_omr.jpeg")
    main()